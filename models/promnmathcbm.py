
# ! CBM model for MN-MATH
import torch
import torch.nn as nn
from utils.args import *
from utils.conf import get_device
from utils.losses import *
from utils.dpl_loss import MNMATH_DPL
from models.utils.cbm_module import CBMModule


def get_parser() -> ArgumentParser:
    """Returns the parser

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class ProMNMATHCBM(CBMModule):
    """CBM MODEL FOR MNIST"""

    NAME = "promnmathcbm"

    """
    MNIST GENERALIZATION TO EQUATIONS: CBM MODEL with Prototypes.
    """

    def __init__(
        self,
        encoder,
        n_images=8,
        c_split=(),
        args=None,
        model_dict=None,
        n_facts=10,
        nr_classes=2,
    ):
        """Initialize method

        Args:
            self: instance
            encoder (nn.Module): encoder
            n_images (int, default=2): number of images
            c_split: concept splits
            args: command line arguments
            model_dict (default=None): model dictionary
            n_facts (int, default=20): number of concepts
            nr_classes (int, nr_classes): number of classes

        Returns:
            None: This function does not return a value.
        """
        super(ProMNMATHCBM, self).__init__(
            encoder=encoder,
            model_dict=model_dict,
            n_facts=n_facts,
            nr_classes=nr_classes,
        )

        # how many images and explicit split of concepts
        self.n_images = 8
        self.c_split = c_split
        # opt and device
        self.opt = None
        self.device = get_device()

        self.classifier = nn.Sequential(
            # nn.Linear(self.n_facts * self.n_images, self.n_facts * self.n_images), nn.ReLU(),
            nn.Linear(self.n_facts * self.n_images, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, self.nr_classes),
            nn.Sigmoid(),
        )


    # & CBM Forward pass
    def forward(self, x, support_images, support_labels):
        """
        Performs a forward pass through the model for a batch of input images and support set.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_images * 28, 28), representing concatenated images.
            support_images (torch.Tensor): Support set images tensor.
            support_labels (torch.Tensor): Support set labels tensor.
        Returns:
            dict: A dictionary containing:
                - "CS" (torch.Tensor): Concept scores tensor of shape (batch_size, n_images, n_facts).
                - "YS" (torch.Tensor): Model inference results.
                - "pCS" (torch.Tensor): Normalized concept scores.
        Raises:
            AssertionError: If input shapes or number of classes do not match expected values.
        """
        
        assert self.nr_classes == 2, f"Expected number of classes 2, but got {self.nr_classes}"
        cs = []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        
        for i in range(self.n_images):  # loop over each digit in the equation
            assert xs[i].shape == (x.size(0), 1, 28, 28), \
                f"Expected input shape {(x.size(0), 1, 28, 28)}, but got {xs[i].shape}"
            _, _, distances = self.encoder.predict(support_images, support_labels, xs[i], unknown_init=False)
            assert distances.shape == (x.size(0), self.n_facts), \
                f"Expected logits shape {(x.size(0), self.n_facts)}, but got {distances.shape}"
            cs.append(distances)
        
        clen = len(cs[0].shape)
        cs = torch.stack(cs, dim=1) if clen == 2 else torch.cat(cs, dim=1)
        assert cs.shape == (x.size(0), self.n_images, self.n_facts), \
            f"Expected shape {(x.size(0), self.n_images, self.n_facts)}, but got {cs.shape}"
        
        pCs = self.normalize_concepts(cs)

        # get the result of the inference
        py = self.cmb_inference(cs)  # cs

        return {"CS": cs, "YS": py, "pCS": pCs}


    # & CBM Inference method
    def cmb_inference(self, cs, query=None):
        """Performs inference inference

        Args:
            self: instance
            cs: concepts logits
            query (default=None): query

        Returns:
            query_prob: query probability
        """

        # flatten the cs
        flattened_cs = cs.view(cs.shape[0], cs.shape[1] * cs.shape[2])

        # Pass the flattened input tensor through the classifier
        query_prob = self.classifier(flattened_cs)

        return query_prob


    # & Normalize concepts
    def normalize_concepts(self, z, split=8):
        """Computes the probability for each ProbLog fact given the latent vector z

        Args:
            self: instance
            z (torch.tensor): latents
            split (int, default=2): numbers of split

        Returns:
            vec: normalized concepts
        """
        # List to hold normalized probabilities for each digit
        normalized_probs = []
        
        # Small value to avoid underflow
        eps = 1e-5

        # Iterate over each digit's latent vector
        for i in range(split):
            # Extract the probability for the current digit
            prob_digit = z[:, i, :]

            # Apply softmax to ensure the probabilities sum to 1
            prob_digit = nn.Softmax(dim=1)(prob_digit)
            
            # Add a small epsilon to avoid ProbLog underflow
            prob_digit = prob_digit + eps
            
            # Normalize the probabilities
            with torch.no_grad():
                Z = torch.sum(prob_digit, dim=-1, keepdim=True)
            prob_digit = prob_digit / Z  # Normalization
            
            # Append the normalized probability to the list
            normalized_probs.append(prob_digit)

        # Stack the normalized probabilities along the dimension for digits
        normalized_probs = torch.stack(normalized_probs, dim=1)
        return normalized_probs


    # & Loss function
    @staticmethod
    def get_loss(args):
        """Loss function for the architecture

        Args:
            args: command line arguments

        Returns:
            loss: loss function

        Raises:
            err: NotImplementedError if the loss function is not available
        """
        if args.dataset in ["mnmath"]:
            return MNMATH_DPL(MNMATH_Cumulative)
        else:
            return NotImplementedError("Wrong dataset choice")