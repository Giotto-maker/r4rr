import torch
import torch.nn as nn

from utils.args import *
from utils.conf import get_device
from utils.losses import *
from models.cext import CExt
from models.utils.utils_problog import build_worlds_queries_matrix
from utils.semantic_loss import ADDMNIST_SL


# * Utils methods: parser method
def get_parser() -> ArgumentParser:
    """Returns the argument parser for this architecture

    Returns:
        argpars: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


# ! Main models Class
class PROMnistSL(CExt):
    """MNIST architecture with SL and Prototypical Networks"""

    NAME = "promnistsl"
    
    # & Initialize the model (OK)
    def __init__(
        self, encoder, n_images=2, c_split=(), args=None, n_facts=20, nr_classes=19
    ):
        """Initialize method

        Args:
            self: instance
            encoder (nn.Module): encoder network
            n_images (int, default=2): number of images
            c_split: concept split
            args (default=None): command line arguments
            n_facts (int, default=20): number of concepts
            nr_classes (int, default=19): number of classes

        Returns:
            None: This function does not return a value.
        """

        super(PROMnistSL, self).__init__(
            encoder=encoder, n_images=n_images, c_split=c_split
        )

        #  Worlds-queries matrix
        if args.task == "addition":
            self.n_facts = (
                10 if not args.dataset in ["halfmnist", "restrictedmnist"] else 5
            )
            self.logic = build_worlds_queries_matrix(2, self.n_facts, "addmnist")
            self.nr_classes = 19
        elif args.task == "product":
            self.n_facts = (
                10 if not args.dataset in ["halfmnist", "restrictedmnist"] else 5
            )
            self.logic = build_worlds_queries_matrix(2, self.n_facts, "productmnist")
            self.nr_classes = 37
        elif args.task == "multiop":
            self.n_facts = 5
            self.logic = build_worlds_queries_matrix(2, self.n_facts, "multiopmnist")
            self.nr_classes = 3
        
        # opt and device
        self.opt = None
        self.device = get_device()
        self.logic = self.logic.to(self.device)

        if args.mlp:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.n_facts * 2, 50),
                torch.nn.ReLU(),
                torch.nn.Linear(50, 50),
                torch.nn.ReLU(),
                torch.nn.Linear(50, self.nr_classes),
            )
        else:
            self.mlp = None

        print("[PROTO-INFO] Prototypical Semantic Loss model initialized successfully.")


    # & Forward pass of the main model to compute the final label using the backbone's raw scores (OK)
    def forward(self, x, support_images, support_labels, unknown_init=False): 
        """
        Forward pass for the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_images * 28, 28) representing concatenated images.
            support_images (torch.Tensor): Support set images used for few-shot prediction.
            support_labels (torch.Tensor): Support set labels corresponding to support_images.
            unknown_init (bool, optional): Flag indicating whether to initialize with unknown class. Defaults to False.
        Returns:
            dict: A dictionary containing:
                - "CS" (torch.Tensor): Raw concept logits of shape (batch_size, n_images, n_facts).
                - "YS" (torch.Tensor): One-hot encoded sum predictions of shape (batch_size, nr_classes).
                - "pCS" (torch.Tensor): Normalized concept probabilities of shape (batch_size, n_images, n_facts).
        Raises:
            AssertionError: If input shapes or intermediate results do not match expected dimensions.
            AssertionError: If predicted digits and their sum are inconsistent.
        """
        
        assert self.nr_classes == 19, f"Expected number of classes 19, but got {self.nr_classes}"
        cs = []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        for i in range(self.n_images):
            assert xs[i].shape == (x.size(0), 1, 28, 28), f"Expected input shape {(x.size(0), 1, 28, 28)}, but got {xs[i].shape}"
            y_hat, log_p_y, distances = self.encoder.predict(support_images, support_labels, xs[i], unknown_init=unknown_init)
            assert distances.shape == (x.size(0), self.n_facts), f"Expected logits shape {(x.size(0), self.n_facts)}, but got {distances.shape}"
            cs.append(distances)
            unknown_init = False  # Reset unknown_init for the next image
            
        cs = torch.stack(cs, dim=1)
        pCs = self.normalize_concepts(cs)
        assert cs.shape == (x.size(0), self.n_images, self.n_facts), \
                f"Expected logits shape {(x.size(0), self.n_images, self.n_facts)}, but got {cs.shape}"
        assert pCs.shape == (x.size(0), self.n_images, self.n_facts), \
                f"Expected distances shape {(x.size(0), self.n_images, self.n_facts)}, but got {pCs.shape}"    

        if self.mlp is None:
            predicted_classes = pCs.argmax(dim=-1)
            
            # Compute the sum directly using the predicted digits
            predicted_digit1, predicted_digit2 = predicted_classes[:, 0], predicted_classes[:, 1]
            predicted_sums = predicted_digit1 + predicted_digit2  # This enforces consistency

            # Create sum probabilities
            sum_probs = torch.zeros(x.size(0), self.nr_classes, device=x.device)
            sum_probs.scatter_(1, predicted_sums.unsqueeze(-1), 1.0)

            for i in range(x.size(0)):
                # Assert consistency between predicted digits and sums
                assert predicted_digit1[i] + predicted_digit2[i] == predicted_sums[i], \
                    f"Inconsistent prediction: digits {predicted_digit1[i].item()}, {predicted_digit2[i].item()} => sum {predicted_sums[i].item()}"
        
            assert sum_probs.shape == (x.size(0), self.nr_classes), \
                    f"Expected sum probabilities shape {(x.size(0), self.nr_classes)}, but got {sum_probs.shape}"
        
            final_predictions = sum_probs
        else:
            pred = self.mlp(cs.view(-1, self.n_facts * 2))
            assert pred.shape == (x.shape[0], self.nr_classes),\
                f"Expected shape {(x.shape[0], self.nr_classes)}, but got {pred.shape}"
            
            final_predictions = pred
        

        return {"CS": cs, "YS": final_predictions, "pCS": pCs}


    # & Returns the semantic loss function for this architecture (OK)
    def get_loss(self, args):
        """Returns the loss function for this architecture

        Args:
            self: instance
            args: command line arguments

        Returns:
            loss: loss function

        Raises:
            err: NotImplementedError if the loss function is not available
        """
        # if args.debug:  print("Getting the semantic loss function for the prototypical network...")
        if args.dataset in ["addmnist", "shortmnist", "restrictedmnist", "halfmnist"]:
            return ADDMNIST_SL(ADDMNIST_Cumulative, self.logic, args)
        else:
            return NotImplementedError("Wrong dataset choice")


    # & Initializes the optimizer for this architecture [CHANGED self.paramters in self.encoder.parameters()] (OK)
    def start_optim(self, args):
        """Initializes the optimizer for this architecture

        Args:
            self: instance
            args: command line arguments

        Returns:
            None: This function does not return a value.
        """
        if args.debug: print("Initializing the optimizer for the prototypical network...")

        if self.mlp is not None:
            self.opt = torch.optim.Adam(
                self.parameters(), args.lr, weight_decay=args.weight_decay
            )
        else:
            self.opt = torch.optim.Adam(
                self.encoder.parameters(), args.lr, weight_decay=args.weight_decay
            )


    # override
    def to(self, device):
        super().to(device)
        self.logic = self.logic.to(device)


    # & Normalize the concepts predictions (OK)
    def normalize_concepts(self, z, split=2):
        """Computes the probability for each ProbLog fact given the latent vector z

        Args:
            self: instance
            z (torch.tensor): latent vector
            split (int, default=2): number of splits

        Returns:
            vec: normalized concept probability
        """
        # Extract probs for each digit

        prob_digit1, prob_digit2 = z[:, 0, :], z[:, 1, :]

        # Sotfmax on digits_probs (the 10 digits values are mutually exclusive)
        prob_digit1 = torch.nn.Softmax(dim=1)(prob_digit1)
        prob_digit2 = torch.nn.Softmax(dim=1)(prob_digit2)

        # Clamp digits_probs to avoid underflow
        eps = 1e-5
        prob_digit1 = prob_digit1 + eps
        with torch.no_grad():
            Z1 = torch.sum(prob_digit1, dim=-1, keepdim=True)
        prob_digit1 = prob_digit1 / Z1  # Normalization
        prob_digit2 = prob_digit2 + eps
        with torch.no_grad():
            Z2 = torch.sum(prob_digit2, dim=-1, keepdim=True)
        prob_digit2 = prob_digit2 / Z2  # Normalization

        return torch.stack([prob_digit1, prob_digit2], dim=1).view(-1, 2, self.n_facts)