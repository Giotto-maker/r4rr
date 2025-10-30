
# ! DPL + PNet model for MN-MATH 
import torch
from models.utils.deepproblog_modules import DeepProblogModel
from utils.args import *
from utils.conf import get_device
from models.utils.utils_problog import *
from utils.losses import MNMATH_Cumulative
from utils.dpl_loss import MNMATH_DPL
from models.utils.ops import outer_product


def get_parser() -> ArgumentParser:
    """Returns the parser

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class ProMNMATHDPL(DeepProblogModel):
    """Prototypical DPL MODEL FOR MN MATH"""

    NAME = "promnmathdpl"

    """
    MATH MNIST Model with Deep Problog Logic (DPL) and Prototypical Network (PNet) backbone.
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
            n_images (int, default=1): number of images
            c_split: concept splits
            args: command line arguments
            model_dict (default=None): model dictionary
            n_facts (int, default=21): number of concepts
            nr_classes (int, nr_classes): number of classes for the multiclass classification problem
            retun_embeddings (bool): whether to return embeddings

        Returns:
            None: This function does not return a value.
        """
        super(ProMNMATHDPL, self).__init__(
            encoder=encoder,
            model_dict=model_dict,
            n_facts=n_facts,
            nr_classes=nr_classes,
        )
        # device
        self.device = get_device()
        self.n_facts = n_facts

        # how many images and explicit split of concepts
        self.c_split = c_split
        self.args = args
        self.n_images = 8

        # logic
        logic_sum = create_mnmath_sum(n_digits=10, sequence_len=4)
        logic_and = create_mnmath_prod(n_digits=10, sequence_len=4)
        logic_combine = create_mnist_and()
        
        # and, sum, combine
        self.logic_sum = logic_sum.to(self.device)
        self.logic_and = logic_and.to(self.device)
        self.combine = logic_combine.to(self.device)

        # opt and device
        self.opt = None


    # & Inference method using PNet encoder
    def forward(self, x, support_images, support_labels):
        """
        Performs a forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor containing concatenated images, expected shape (batch_size, n_images * 28, 28).
            support_images (torch.Tensor): Support set images for few-shot learning, shape depends on encoder requirements.
            support_labels (torch.Tensor): Labels corresponding to support_images.
        Returns:
            dict: A dictionary containing:
                - "CS" (torch.Tensor): Raw concept logits, shape (batch_size, n_images, n_facts).
                - "YS" (torch.Tensor): Output probabilities from Problog inference.
                - "pCS" (torch.Tensor): Normalized concept predictions.
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
        
        # ^ recenter the logits around 0 as DPL works best in that range
        cs = cs - cs.mean()

        # normalize concept preditions
        pCs = self.normalize_concepts(cs)

        # Problog inference to compute worlds and query probability distributions
        py, worlds_prob = self.problog_inference(pCs)

        return {"CS": cs, "YS": py, "pCS": pCs}


    # & Inference method to combine the worlds probability distribution P(w) and compute the query probability P(q)
    def problog_inference(self, pCs, query=None):
        """Performs ProbLog inference to retrieve the worlds probability distribution P(w).
        Works with an arbitrary number of encoded bits (digits).

        Args:
            self: instance
            pCs: probability of concepts (shape: [batch_size, num_digits, num_classes])
            query (default=None): query

        Returns:
            query_prob: query probability
            worlds_prob: worlds probability
        """

        # Extract digit probability
        prob_digit1, prob_digit2, prob_digit3, prob_digit4 = pCs[:, 0, :], pCs[:, 1, :], pCs[:, 2, :], pCs[:, 3, :]

        # Extract again digit probability
        prob_digit5, prob_digit6, prob_digit7, prob_digit8 = pCs[:, 4, :], pCs[:, 5, :], pCs[:, 6, :], pCs[:, 7, :]

        # Compute worlds probability P(w) for sum
        probs_for_sum = outer_product(prob_digit1, prob_digit2, prob_digit3, prob_digit4)
        worlds_prob_sum = probs_for_sum.reshape(-1, int(self.n_facts ** (self.n_images / 2)))

        probs_for_prod = outer_product(prob_digit5, prob_digit6, prob_digit7, prob_digit8)
        worlds_prob_prod = probs_for_prod.reshape(-1, int(self.n_facts ** (self.n_images / 2)))

        # Compute query probability P(q)
        query_prob_sum = torch.zeros(
            size=(len(worlds_prob_sum), self.nr_classes), device=probs_for_sum.device
        )
        query_prob_prod = torch.zeros(
            size=(len(worlds_prob_prod), self.nr_classes), device=probs_for_prod.device
        )

        for i in range(self.nr_classes):
            query = i
            query_prob_sum[:, i] = self.compute_query_sum(query, worlds_prob_sum).view(-1)

        for i in range(self.nr_classes):
            query = i
            query_prob_prod[:, i] = self.compute_query_prod(query, worlds_prob_prod).view(-1)

        # add a small offset
        query_prob_prod += 1e-5
        with torch.no_grad():
            Z = torch.sum(query_prob_prod, dim=-1, keepdim=True)
        query_prob_prod = query_prob_prod / Z

        # add a small offset
        query_prob_sum += 1e-5
        with torch.no_grad():
            Z = torch.sum(query_prob_sum, dim=-1, keepdim=True)
        query_prob_sum = query_prob_sum / Z

        combined_tensor = torch.stack((query_prob_sum[:, 1], query_prob_prod[:, 1]), dim=1)
        
        return combined_tensor, None


    # & Compute query probability P(q) for sum (first label) given the worlds probability P(w)
    def compute_query_sum(self, query, worlds_prob):
        """Computes query probability given the worlds probability P(w).

        Args:
            self: instance
            query: query
            worlds_probs: worlds probabilities

        Returns:
            query_prob: query probabilities
        """
        # Select the column of w_q matrix corresponding to the current query
        w_q = self.logic_sum[:, query]

        # Compute query probability by summing the probability of all the worlds where the query is true
        query_prob = torch.sum(w_q * worlds_prob, dim=1, keepdim=True)
        return query_prob


    # & Compute query probability P(q) for product (second label) given the worlds probability P(w)
    def compute_query_prod(self, query, worlds_prob):
        """Computes query probability given the worlds probability P(w).

        Args:
            self: instance
            query: query
            worlds_probs: worlds probabilities

        Returns:
            query_prob: query probabilities
        """
        # Select the column of w_q matrix corresponding to the current query
        w_q = self.logic_and[:, query]

        # Compute query probability by summing the probability of all the worlds where the query is true
        query_prob = torch.sum(w_q * worlds_prob, dim=1, keepdim=True)
        return query_prob


    # & Compute query probability P(q) for combined label given the worlds probability P(w)
    def compute_query_combine(self, query, worlds_prob):
        """Computes query probability given the worlds probability P(w).

        Args:
            self: instance
            query: query
            worlds_probs: worlds probabilities

        Returns:
            query_prob: query probabilities
        """
        # Select the column of w_q matrix corresponding to the current query
        w_q = self.combine[:, query]
        # Compute query probability by summing the probability of all the worlds where the query is true
        query_prob = torch.sum(w_q * worlds_prob, dim=1, keepdim=True)
        return query_prob


    # & Normalize the concepts probabilities
    def normalize_concepts(self, z, split=8):
        """Computes the probability for each ProbLog fact given the latent vector z

        Args:
            self: instance
            z (torch.tensor): latents
            split (int, default=2): number of splits (number of digits)

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


    # & Get the loss function for the architecture
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


    # override
    def to(self, device):
        super().to(device)
        self.logic_and = self.logic_and.to(device)
        self.logic_sum = self.logic_sum.to(device)
        self.combine = self.combine.to(device)