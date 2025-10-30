import torch
from torch.nn.functional import one_hot
from utils.kand_ltn_loss import KAND_SAT_AGG
from utils.conf import get_device
from utils.args import *
import ltn


# from https://unitn-sml.github.io/rsbench/
def get_parser() -> ArgumentParser:
    """Returns the argument parser for the current architecture

    Returns:
        argpars: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class KANDltn(torch.nn.Module):
    """LTN architecture for Kandinsky"""

    NAME = "kandltn"
    """
    Kandinsky patterns
    """

    def __init__(self, encoder, n_images=3, c_split=(), args=None):
        super(KANDltn, self).__init__()
        """Initialize method

        Args:
            self: instance
            encoder (nn.Module): encoder
            n_images (int, default=2): number of images
            c_split: concept splits
            args: command line arguments

        Returns:
            None: This function does not return a value.
        """
        # bones of the model
        self.encoder = encoder
        print(self.encoder)

        # number of images, and how to split them
        self.n_images = n_images

        # opt and device
        self.opt = None
        self.device = get_device()

    def forward(self, x):
        """Forward method

        Args:
            self: instance
            x (torch.tensor): input vector

        Returns:
            out_dict: output dictionary
        """
        cs = []
        xs = torch.split(
            x, x.size(-1) // self.n_images, dim=-1
        )  # split the tensor in three tensors containing the three images
        for i in range(
            self.n_images
        ):  # for each image, call the encoder to get the concept related to that image

            lc, _ = self.encoder(
                xs[i]
            )  # this will return 6 concepts for each shape included in the image
            cs.append(lc)
        
        cs = torch.stack(cs, dim=1)
            
        # reshape in such a way the figures are separated
        res_cs = torch.reshape(
            cs, (cs.shape[0], 3, 3, 6)
        )  # b_size X #_figures X #_shapes X #_concepts

        # split shape concepts from color concepts
        cs_split = torch.split(res_cs, 3, dim=-1)
        shape_logits = cs_split[0]
        color_logits = cs_split[1]

        # apply softmax to simulate multi-class classification of shape and color
        shape_probs = torch.nn.Softmax(dim=-1)(shape_logits)
        color_probs = torch.nn.Softmax(dim=-1)(color_logits)

        # get predictions
        shape_preds = torch.argmax(shape_probs, dim=-1)
        color_preds = torch.argmax(color_probs, dim=-1)

        # compute final predictions for accuracy computation
        with torch.no_grad():
            # Define lambda functions for "all same" and "all different"
            same = lambda inp: torch.all(inp == inp[:, :, 0].unsqueeze(-1), dim=2, keepdim=True)
            diff = lambda inp: (torch.tensor([len(torch.unique(fig)) for sample in inp for fig in sample])
                                == 3).view(inp.shape[0], inp.shape[1], 1)
            # For shapes:
            same_s_tensor = same(shape_preds.detach().cpu())
            diff_s_tensor = diff(shape_preds.detach().cpu())
            # Pair predicate: not all same and not all different.
            pair_s_tensor = ~(same_s_tensor | diff_s_tensor)
            
            # For colors:
            same_c_tensor = same(color_preds.detach().cpu())
            diff_c_tensor = diff(color_preds.detach().cpu())
            pair_c_tensor = ~(same_c_tensor | diff_c_tensor)
            
            # Aggregate over images (each has shape [batch, 3, 1], so we check if all images satisfy the condition)
            same_s = torch.all(same_s_tensor, dim=1)
            diff_s = torch.all(diff_s_tensor, dim=1)
            pair_s = torch.all(pair_s_tensor, dim=1)
            same_c = torch.all(same_c_tensor, dim=1)
            diff_c = torch.all(diff_c_tensor, dim=1)
            pair_c = torch.all(pair_c_tensor, dim=1)
            
            # Final prediction: pattern holds if any of the predicates holds for either shapes or colors.
            preds = torch.any(torch.cat([same_s, diff_s, pair_s, same_c, diff_c, pair_c], dim=1), dim=1).long()

        return {
            "CS": cs,
            "pCS": torch.cat([shape_probs, color_probs], dim=-1),
            "YS": one_hot(preds.view(-1), num_classes=2),
        }

    def get_loss(self, args):
        """Returns the loss function for the architecture

        Args:
            self: instance
            args: command line arguments

        Returns:
            loss: loss function

        Raises:
            err: NotImplementedError if loss is not implemented
        """
        _and, _implies = None, None
        if args.and_op == "Godel":
            _and = ltn.fuzzy_ops.AndMin()
        elif args.and_op == "Prod":
            _and = ltn.fuzzy_ops.AndProd()
        else:
            _and = ltn.fuzzy_ops.AndLuk()
        if args.or_op == "Godel":
            Or = ltn.Connective(ltn.fuzzy_ops.OrMax())
        elif args.or_op == "Prod":
            Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
        else:
            Or = ltn.Connective(ltn.fuzzy_ops.OrLuk())
        if args.imp_op == "Godel":
            _implies = ltn.fuzzy_ops.ImpliesGodel()
        elif args.imp_op == "Prod":
            _implies = ltn.fuzzy_ops.ImpliesReichenbach()
        elif args.imp_op == "Luk":
            _implies = ltn.fuzzy_ops.ImpliesLuk()
        elif args.imp_op == "Goguen":
            _implies = ltn.fuzzy_ops.ImpliesGoguen()
        else:
            _implies = ltn.fuzzy_ops.ImpliesKleeneDienes()

        And = ltn.Connective(_and)
        Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        Forall = ltn.Quantifier(
            ltn.fuzzy_ops.AggregPMeanError(p=args.p), quantifier="f"
        )
        Equiv = ltn.Connective(ltn.fuzzy_ops.Equiv(and_op=_and, implies_op=_implies))
        return KAND_SAT_AGG(And, Or, Not, Forall, Equiv)

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
