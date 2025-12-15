class Layer:
    def forward(self, input):
        """
        Forward pass.
        """
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        """
        Backward pass.
        """
        raise NotImplementedError