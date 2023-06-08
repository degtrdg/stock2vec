import torch
import torch.nn as nn
import torch.nn.functional as F


class Time2Vec(nn.Module):
    def __init__(self, kernel_size=1):
        super(Time2Vec, self).__init__()
        self.kernel_size = kernel_size
        self.weights_initialized = False

    def forward(self, inputs):
        """Forward pass of the Time2Vec module"""
        # inputs is expected to be [batch_size, sequence_length, feature_dim]

        if not self.weights_initialized:
            self.initialize_weights(inputs.shape)
            self.weights_initialized = True

        # Linear transformation
        linear_transformation = self.weight_bias * inputs + self.bias_bias
        # Angular transformation
        # We add an extra dimension for inputs to match the shape of weight_angle
        # input shape becomes [batch_size, sequence_length, feature_dim, 1]
        inputs = inputs.unsqueeze(-1)
        # expand weight_angle shape to [batch_size, sequence_length, kernel_size]
        # expand weight_angle shape to [batch_size, sequence_length, kernel_size, 1]
        weight_angle = self.weight_angle.unsqueeze(
            -1).expand(-1, -1, -1, inputs.shape[0])
        angular_transformation = torch.matmul(
            inputs, weight_angle) + self.bias_angle
        angular_transformation = torch.sin(angular_transformation)

        # Combine linear and angular transformations
        # Then reshape it to a 2D tensor
        combined = torch.cat(
            [linear_transformation.unsqueeze(-1), angular_transformation], -1)
        output = combined.view(-1, inputs.shape[1] * (self.kernel_size + 1))

        # output is expected to be [batch_size, sequence_length * (kernel_size + 1)]
        return output

    def initialize_weights(self, input_shape):
        """Initializes the weights and biases parameters"""
        # input_shape is expected to be [batch_size, sequence_length, feature_dim]

        # Initialize weights and bias for linear transformation (wb and bb respectively)
        self.weight_bias = nn.Parameter(
            torch.randn(input_shape[1], input_shape[2]))
        self.bias_bias = nn.Parameter(
            torch.randn(input_shape[1], input_shape[2]))

        # Initialize weights and bias for angular transformation (wa and ba respectively)
        self.weight_angle = nn.Parameter(
            torch.randn(1, input_shape[1], self.kernel_size))
        self.bias_angle = nn.Parameter(
            torch.randn(1, input_shape[1], self.kernel_size))


class AttentionBlock(nn.Module):
    def __init__(self, num_heads=2, head_size=128, ff_dim=None, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        if ff_dim is None:
            ff_dim = head_size

        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=head_size, num_heads=num_heads, dropout=dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(
            normalized_shape=head_size, eps=1e-6)

        # Feed forward layer
        self.feed_forward_conv1 = nn.Conv1d(
            in_channels=head_size, out_channels=ff_dim, kernel_size=1)
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.feed_forward_norm = nn.LayerNorm(
            normalized_shape=head_size, eps=1e-6)

    def build(self, input_shape):
        # We only know the input_shape during the forward pass, so we define the second convolution layer here
        self.feed_forward_conv2 = nn.Conv1d(
            in_channels=self.feed_forward_conv1.out_channels, out_channels=input_shape[-1], kernel_size=1)

    def forward(self, inputs):
        # inputs is expected to be [batch_size, sequence_length, embedding_dim]
        self.build(inputs.shape)

        # Permute the inputs for multi-head attention
        # [sequence_length, batch_size, embedding_dim]
        inputs = inputs.permute(1, 0, 2)

        # Multi-head attention
        attention_output, _ = self.attention(inputs, inputs, inputs)
        attention_output = self.attention_dropout(attention_output)
        attention_output = self.attention_norm(inputs + attention_output)

        # Permute back for feed forward network
        # [batch_size, embedding_dim, sequence_length]
        attention_output = attention_output.permute(1, 2, 0)

        # Feed forward network
        ff_output = F.relu(self.feed_forward_conv1(attention_output))
        ff_output = self.feed_forward_conv2(ff_output)
        ff_output = self.feed_forward_dropout(ff_output)

        # Permute back for layer normalization
        # [sequence_length, batch_size, embedding_dim]
        ff_output = ff_output.permute(2, 0, 1)

        # Layer normalization
        output = self.feed_forward_norm(inputs + ff_output)

        # Permute back for the final output
        # [batch_size, sequence_length, embedding_dim]
        output = output.permute(1, 0, 2)

        return output


class ModelTrunk(nn.Module):
    def __init__(self, time2vec_dim=1, num_heads=2, head_size=128, ff_dim=None, num_layers=1, dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.time2vec = Time2Vec(kernel_size=time2vec_dim)
        if ff_dim is None:
            ff_dim = head_size

        # We create multiple attention blocks
        self.attention_layers = nn.ModuleList([AttentionBlock(
            num_heads=num_heads, head_size=head_size, ff_dim=ff_dim, dropout=dropout) for _ in range(num_layers)])

    def forward(self, inputs):
        # inputs is expected to be [batch_size, sequence_length, feature_dim]

        # Get time embeddings
        time_embedding = self.time2vec(inputs)

        # Concatenate the inputs and time embeddings
        # [batch_size, sequence_length, feature_dim + time2vec_dim]
        x = torch.cat([inputs, time_embedding], -1)

        # Pass through attention layers
        for attention_layer in self.attention_layers:
            # [batch_size, sequence_length, embedding_dim]
            x = attention_layer(x)

        # Reshape output to be 2D (flatten the last two dimensions)
        # [batch_size, sequence_length * embedding_dim]
        output = x.view(-1, x.shape[1] * x.shape[2])

        return output
