import torch

from utils.args import *
from utils.losses import *
from utils.conf import get_device
from utils.dpl_loss import ADDMNIST_DPL

from pishield.shield_layer import build_shield_layer

from models.cext import CExt


# * Argument parser class
def get_parser() -> ArgumentParser:
    """Returns the argument parser for this architecture

    Returns:
        argpars: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class ProShieldedMNIST(CExt):
    """MNIST architecture with CCN+"""

    NAME = "proshieldedmnist"
    
    # * Intiliaze Prototypical CCN+ model
    def __init__(
        self,
        encoder, 
        n_images=2, 
        c_split=(), 
        args=None, 
        n_facts=10, 
        nr_classes=19
    ):
        super(ProShieldedMNIST, self).__init__(
            encoder=encoder, n_images=n_images, c_split=c_split
        )
        self.n_facts = n_facts
        self.nr_classes = nr_classes
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.n_facts * 2, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, self.nr_classes),
        )
        self._shield_layer = None
        self.opt = None
        self.device = get_device()
            
    
    # * Forward pass with prototypes
    def forward(self, x, support_images, support_labels, unknown_init=False):
        cs = []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        
        # ^ compute the logits for each digit (disentanglement)
        for i in range(self.n_images):
            assert xs[i].shape == (x.size(0), 1, 28, 28), f"Expected input shape {(x.size(0), 1, 28, 28)}, but got {xs[i].shape}"
            y_hat, log_p_y, distances = self.encoder.predict(support_images, support_labels, xs[i], unknown_init=unknown_init)
            assert distances.shape == (x.size(0), self.n_facts), f"Expected logits shape {(x.size(0), self.n_facts)}, but got {distances.shape}"
            cs.append(distances)
            unknown_init = False  # Reset unknown_init for the next image
        clen = len(cs[0].shape)
        
        # ^ combine the logits for both digits
        cs = torch.stack(cs, dim=1) if clen == 2 else torch.cat(cs, dim=1)  
        assert cs.shape == (x.shape[0], 2, self.n_facts),\
            f"Expected shape {(x.shape[0], 2, self.n_facts)}, but got {cs.shape}"
        
        # ^ normalize concept predictions for each digit
        pCs = self.normalize_concepts(cs)

        # ^ predict the final label based on the two digits
        pred = self.mlp(cs.view(-1, self.n_facts * 2))
        assert pred.shape == (x.shape[0], self.nr_classes),\
            f"Expected shape {(x.shape[0], self.nr_classes)}, but got {pred.shape}"
        
        sum_probs = torch.softmax(pred, dim=1)

        # ^ combine concept probabilities and final prediction probabilities
        pCs_flat = pCs.reshape(pCs.shape[0], -1)
        assert pCs_flat.shape == (x.shape[0], self.n_facts * 2),\
            f"Expected shape {(x.shape[0], self.n_facts * 2)}, but got {pCs_flat.shape}"
        combined_predictions = torch.cat([pCs_flat, sum_probs], dim=1)
        assert combined_predictions.shape == (x.shape[0], self.n_facts * 2 + self.nr_classes),\
            f"Expected shape {(x.shape[0], self.n_facts * 2 + self.nr_classes)}, but got {combined_predictions.shape}"
        
        # ^ correct the predictions and renormalize them
        corrected_combined = self.shield_layer(combined_predictions)
        correct_final = self.renormalize_probabilities(corrected_combined[:, self.n_facts * 2:], dim=1)
        assert corrected_combined.shape == (x.shape[0], self.n_facts * 2 + self.nr_classes),\
            f"Expected shape {(x.shape[0], self.n_facts * 2 + self.nr_classes)}, but got {corrected_combined.shape}"
        assert correct_final.shape == (x.shape[0], self.nr_classes),\
            f"Expected shape {(x.shape[0], self.nr_classes)}, but got {correct_final.shape}"
        
        return {"CS": cs, "YS": correct_final, "pCS": pCs}
        
    # * Normalize concept predictions
    def normalize_concepts(self, z, split=2):
        """Computes the probability for each fact given the latent vector z
        Args:
            self: instance
            z (torch.tensor): latent vector
            split (int, default=2): number of splits

        Returns:
            vec: normalized concept probability
        """
        prob_digit1, prob_digit2 = z[:, 0, :], z[:, 1, :]
        prob_digit1 = torch.nn.Softmax(dim=1)(prob_digit1)
        prob_digit2 = torch.nn.Softmax(dim=1)(prob_digit2)
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
        
    # * Get loss function for training
    def get_loss(self, args):
        """Loss function for the architecture
        Args:
            args: command line arguments
        Returns:
            loss: loss function
        Raises:
            err: NotImplementedError if the loss function is not available
        """
        if args.dataset in ["addmnist", "shortmnist", "restrictedmnist", "halfmnist"]:
            return ADDMNIST_DPL(ADDMNIST_Cumulative)
        else:
            return NotImplementedError("Wrong dataset choice")
        
    @property
    def shield_layer(self):
        if self._shield_layer is None:
            custom_ordering = ",".join(str(i) for i in range((self.n_facts * 2) + self.nr_classes))
            print(f"Custom ordering for shield: {custom_ordering}")
            self._shield_layer = build_shield_layer(
                (self.n_facts * 2) + self.nr_classes,
                'CCN+/mnist_EvenOdd.txt',
                ordering_choice='custom',
                custom_ordering=custom_ordering
            ).to(self.device)
        return self._shield_layer


    # * Initialize optimizer
    def start_optim(self, args):
        """Initializes the optimizer for this architecture

        Args:
            self: instance
            args: command line arguments

        Returns:
            None: This function does not return a value.
        """
        self.opt = torch.optim.Adam(
            self.parameters(), args.lr, weight_decay=args.weight_decay
        )

    # * Renormalize probabilities to sum to 1
    @staticmethod
    def renormalize_probabilities(probs, dim=-1, eps=1e-8):
        """Renormalize probabilities to sum to 1."""
        return probs / (probs.sum(dim=dim, keepdim=True) + eps)

    # override
    def to(self, device):
        super().to(device)