import torch
from utils.args import *
from utils.conf import get_device
from utils.losses import *
from models.cext import CExt
from models.utils.utils_problog import build_worlds_queries_matrix_KAND
from utils.semantic_loss import KANDINSKY_SL

def get_parser() -> ArgumentParser:
    """Returns the argument parser for this architecture"""
    parser = ArgumentParser(description="Learning via Kandinsky Concept Extractor.")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class KandSLOneEmbedding(CExt):
    """Kandinsky architecture with SL"""

    NAME = "kandsloneembedding"

    # & Ok 
    def __init__(self, encoder, n_images=3, c_split=(), args=None, n_facts=6, nr_classes=2):
        """
        Initialize the Kandinsky architecture with semantic loss.
        
        Args:
            encoder (nn.Module): Encoder network
            n_images (int): Number of subimages per image (default: 3)
            c_split: Concept split configuration
            args: Command-line arguments
            n_facts (int): Number of concepts (default: 6 for 3 shapes × 2 colors)
            nr_classes (int): Number of output classes (default: 2 for binary classification)
        """
        super(KandSLOneEmbedding, self).__init__(encoder=encoder, n_images=n_images, c_split=c_split)

        # Logical world-query matrix
        self.n_facts = n_facts
        self.logic, _ = build_worlds_queries_matrix_KAND(
            n_images=self.n_images, 
            n_concepts=self.n_facts, 
            n_poss=3,
            task="patterns")
        self.nr_classes = nr_classes

        assert self.n_images == 3, "KandinskySL expects 3 subimages"
        assert self.n_facts == 6, "KandinskySL expects 3 shapes × 2 colors"
        assert self.nr_classes == 2, "KandinskySL only supports binary classification"
        assert self.logic.shape == (729, 9), f"Expected shape (729, 9), but got {self.logic.shape}"

        # Optimizer and device configuration
        self.opt = None
        self.device = get_device()
        self.logic = self.logic.to(self.device)


    # & Forward method
    def forward(self, x):
        """Forward method for the model.
        
        Args:
            x (torch.tensor): Input tensor with concatenated subimages
            
        Returns:
            out_dict: Dictionary of model predictions
        """
        # Encode subimages
        cs = []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        for i in range(self.n_images):
            # Latent concepts from encoder
            lc, _ = self.encoder(xs[i])
            cs.append(lc)
        clen = len(cs[0].shape)
        cs = torch.stack(cs, dim=1) if clen == 2 else torch.cat(cs, dim=1) # torch.Size([32, 3, 18])
        
        # Normalize concept predictions
        pCs = self.normalize_concepts(cs)
        
        # Aggregate concepts across batch size to get predictions
        batch_size = x.size(0)
        YS = self.aggregate_concepts(pCs, x)

        # For consistency, one may include assertions on the shapes:
        assert cs.shape[0] == batch_size and cs.shape[1] == self.n_images, "Unexpected cs shape"
        assert pCs.shape == (batch_size, self.n_images, 3, 2, 3), f"Expected pCs shape {(batch_size, self.n_images, 3, 2, 3)}, got {pCs.shape}"
        assert YS.shape == (batch_size, self.nr_classes), f"Expected YS shape {(batch_size, self.nr_classes)}, got {YS.shape}"

        # Return predictions
        return {"CS": cs, "YS": YS, "pCS": pCs}
    

    # & Symbolic reasoner to get the predictions from the atomic concepts predictions
    def aggregate_concepts(self, pCs, x): 
        pred_shape = pCs[:, :, :, 0, :].argmax(dim=-1)  # [batch, n_images, 3]
        pred_color = pCs[:, :, :, 1, :].argmax(dim=-1)  # [batch, n_images, 3]

        batch_size = x.size(0)
        def all_different(vec):     return (vec.unique().numel() == 3)
        def all_same(vec):          return (vec.unique().numel() == 1)
        def exactly_two_same(vec):  return (vec.unique().numel() == 2)

        pred_labels = torch.zeros(batch_size, dtype=torch.long, device=x.device)

        # Iterate over samples
        for b in range(batch_size):
            # For each subimage, compute predicates for shapes and colors.
            # Each image yields three booleans per predicate.
            shape_preds = []  # list of dicts for each subimage
            color_preds = []
            for i in range(self.n_images):
                # For the current sample and subimage: a tensor of shape [3] for shapes/colors.
                sh = pred_shape[b, i]  # tensor of shape [3]
                co = pred_color[b, i]  # tensor of shape [3]
                shape_preds.append({
                    'diff': all_different(sh),
                    'same': all_same(sh),
                    'two': exactly_two_same(sh)
                })
                color_preds.append({
                    'diff': all_different(co),
                    'same': all_same(co),
                    'two': exactly_two_same(co)
                })
            # For each predicate, check if all images satisfy it.
            # We have six candidates: diff/shapes, diff/colors, two/shapes, two/colors, same/shapes, same/colors.
            cond_diff_color = all(pred['diff'] for pred in color_preds)
            cond_same_color = all(pred['same'] for pred in color_preds)
            cond_two_color  = all(pred['two']  for pred in color_preds)
            cond_diff_shape = all(pred['diff'] for pred in shape_preds)
            cond_same_shape = all(pred['same'] for pred in shape_preds)
            cond_two_shape  = all(pred['two']  for pred in shape_preds)
            
            # If any of the conditions holds, then the pattern is satisfied.
            if cond_diff_color or cond_same_color or cond_two_color or cond_diff_shape or cond_same_shape or cond_two_shape:
                pred_labels[b] = 1  # pattern holds
            else:
                pred_labels[b] = 0  # pattern does not hold

        # --- Create one-hot probabilities ---
        # Initialize with zeros and then scatter 1.0 at the predicted class index.
        YS = torch.zeros(batch_size, self.nr_classes, device=x.device)
        YS.scatter_(1, pred_labels.unsqueeze(-1), 1.0)

        # --- Apply temperature scaling to logits ---
        # Introduce a temperature parameter to control the sharpness of the logits.
        temperature = 0.5  # Adjust this value as needed
        YS = torch.nn.functional.softmax(YS / temperature, dim=-1)

        return YS


    # & Normalize concepts method to get pCs for cs
    def normalize_concepts(self, z):
        """
        Normalize the logits of concepts for shapes and colors.
        
        Args:
            z (torch.tensor): Latent logits for concepts (shape: [batch_size, n_images, 18])
            
        Returns:
            Normalized probabilities for shapes and colors for all figures.
        """
        # Split logits into shapes and colors
        shape_logits = z[:, :, :9]  # First 9 are shapes (3 figures × 3 shapes each)
        color_logits = z[:, :, 9:]  # Last 9 are colors (3 figures × 3 colors each)

        # Reshape to separate logits for each figure
        shape_logits = shape_logits.view(z.size(0), z.size(1), 3, 3)  # [batch_size, n_images, 3 figures, 3 shapes]
        color_logits = color_logits.view(z.size(0), z.size(1), 3, 3)  # [batch_size, n_images, 3 figures, 3 colors]

        # Apply softmax to normalize logits into probabilities
        prob_shape = torch.nn.functional.softmax(shape_logits, dim=-1)  # [batch_size, n_images, 3 figures, 3 shapes]
        prob_color = torch.nn.functional.softmax(color_logits, dim=-1)  # [batch_size, n_images, 3 figures, 3 colors]

        # Return tensor with shape [batch_size, n_images, 3 figures, 2 (shape/color), 3 (concepts)]
        return torch.stack([prob_shape, prob_color], dim=3)


    # & Semantic Loss method
    def get_loss(self, args):
        """Returns the loss function for this architecture.
        
        Args:
            args: Command-line arguments
            
        Returns:
            Kandinsky-specific semantic loss
        """
        if args.dataset in ["kandinsky"]:
            return KANDINSKY_SL(KAND_Cumulative, self.logic, args)
        else:
            raise NotImplementedError("Dataset not supported for KandinskySL")


    def start_optim(self, args):
        """Initializes the optimizer for this architecture.
        
        Args:
            args: Command-line arguments
        """
        self.opt = torch.optim.Adam(
            self.parameters(), args.lr, weight_decay=args.weight_decay
        )


    # Override to ensure logic is on the correct device
    def to(self, device):
        super().to(device)
        self.logic = self.logic.to(device)
