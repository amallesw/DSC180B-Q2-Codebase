class BaseClassifier:
    """
    A base class for classifiers, initializing configuration based on model architecture and classifier type.

    Attributes:
        config (dict): Configuration settings for the model.
        emotion_labels (list[str]): A predefined list of emotion labels.
        model_config (dict): Configuration settings specific to the model architecture and classifier.
        model (Optional[nn.Module]): The model instance, to be defined by subclasses.
    """

    def __init__(self, config: dict, model_architecture: str, classifier: str) -> None:
        """
        Initializes the BaseClassifier with the specified configuration, model architecture, and classifier type.

        Args:
            config (dict): The configuration dictionary containing settings for the model.
            model_architecture (str): The architecture of the model being used. If not specified, falls back to a default.
            classifier (str): The classifier type.
        """
        self.config = config
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        # Directly accessing model configuration based on model_architecture and classifier presence
        if not model_architecture:
            # If model_architecture is not specified, access the top-level classifier configuration
            self.model_config = config["model"]["CNN"]
            self.model_config = config["model"][classifier]
        else:
            # Access configuration based on model_architecture and classifier
            self.model_config = config["model"][model_architecture][classifier]
        self.model = None  # To be instantiated by subclasses

    def create_dataloaders(self, embeddings, labels):
        """Abstract method for creating data loaders. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")

    def train_and_validate(self, train_loader, val_loader, num_epochs):
        """Abstract method for training and validation. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")

    def evaluate_model(self, test_loader):
        """Abstract method for model evaluation. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")
