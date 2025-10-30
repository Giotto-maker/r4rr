import torch
import matplotlib.pyplot as plt

from models.utils.deepproblog_modules import DeepProblogModel
from utils.args import *
from utils.conf import get_device
from models.utils.utils_problog import *
from utils.losses import *
from utils.dpl_loss import KAND_DPL
from models.utils.ops import outer_product

from shortcut_mitigation.kandinsky.protonet_kand_modules.utility_modules.visualizers import plot_primitives
from shortcut_mitigation.kandinsky.protonet_kand_modules.yolo_modules.yolo import yolo_detect_and_crop_primitives_batch

# * Parser method
def get_parser() -> ArgumentParser:
    """Returns the argument parser for this architecture"""
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

# * Utility method to filter out hidden classes
def filter_hidden_classes(images, labels, hide):
    """
    Filters out images and labels corresponding to the classes in the hide list.

    Args:
        images (torch.Tensor): A tensor of shape (batch_size, 3, 64, 64).
        labels (torch.Tensor): A tensor of shape (batch_size) with labels 0, 1, or 2.
        hide (list): A list of integers representing the classes to hide.

    Returns:
        torch.Tensor, torch.Tensor: Filtered images and labels.
    """
    if hide:
        # Create a mask for labels not in the hide list
        mask = ~torch.isin(labels, torch.tensor(hide, device=labels.device))
        
        # Apply the mask to filter images and labels
        filtered_images = images[mask]
        filtered_labels = labels[mask]
        
        return filtered_images, filtered_labels
    else:
        return images, labels


# ! Model Class
class ProKandDPL(DeepProblogModel):
    """Kandinsky DPL model"""
    NAME = "prokanddpl"

    # & Init method
    def __init__(
        self,
        encoder,
        n_images=3,
        c_split=(),
        args=None,
        model_dict=None,
        n_facts=20,
        nr_classes=19,
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
        super(ProKandDPL, self).__init__(
            encoder=encoder,
            model_dict=model_dict,
            n_facts=n_facts,
            nr_classes=nr_classes,
        )

        # ^ register the encoders as a parameter
        self.encoder = nn.ModuleList(encoder)

        self.n_images = n_images
        self.c_split = c_split

        self.n_facts = 6
        self.w_q, self.and_rule = build_worlds_queries_matrix_KAND(
            self.n_images, self.n_facts, 3, task=args.task
        )
        self.n_predicates = 9
        self.nr_classes = 2

        self.opt = None
        self.device = get_device()
        self.w_q = self.w_q.to(self.device)
        self.and_rule = self.and_rule.to(self.device)


    # & Forward method
    def forward(self, x, concept_extractor, transform, support_images, support_labels, args, unknown_init=False):
        
        cs, pCs, preds = [], [], []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)

        for i in range(self.n_images):
            assert xs[i].shape == (x.size(0), 3, 64, 64), f"Expected input shape {(x.size(0), 3, 64, 64)}, but got {xs[i].shape}"
            if args.debug:  plot_primitives(xs[i].cpu(), support_labels[:, 0], num_images=3)

            cropped_objects = yolo_detect_and_crop_primitives_batch(xs[i], concept_extractor, transform, args)

            # ^ hide classes if required
            support_shape_labels = support_labels[:, 0]
            support_color_labels = support_labels[:, 1]
            support_images_shapes, support_shape_labels = filter_hidden_classes(support_images, support_shape_labels, args.hide_shapes)
            support_images_colors, support_color_labels = filter_hidden_classes(support_images, support_color_labels, args.hide_colors)
            if args.hide_shapes: assert not any(label in args.hide_shapes for label in support_shape_labels), "shape_labels contains hidden classes"
            if args.hide_colors: assert not any(label in args.hide_colors for label in support_color_labels), "color_labels contains hidden classes"
            support_shape_labels = support_shape_labels.view(-1, 1)
            support_color_labels = support_color_labels.view(-1, 1)
            assert support_shape_labels.shape == (support_images_shapes.size(0), 1), \
                f"Expected support shape {(support_labels.size(0), 1)}, but got {support_shape_labels.shape}"
            assert support_color_labels.shape == (support_images_colors.size(0), 1), \
                f"Expected support shape {(support_labels.size(0), 1)}, but got {support_color_labels.shape}"
            
            _, _, shapes_logits = self.encoder[0].predict(support_images, support_shape_labels, cropped_objects, unknown_init=False) # ^ (96,3)
            assert shapes_logits.shape == (xs[i].size(0)*3, 3), f"Expected shapes_logits shape {(xs[i].size(0), 3)}, but got {shapes_logits.shape}"
            _, _, colors_logits = self.encoder[1].predict(support_images, support_color_labels, cropped_objects, unknown_init=False) # ^ (96,3)
            assert colors_logits.shape == (xs[i].size(0)*3, 3), f"Expected colors_logits shape {(xs[i].size(0), 3)}, but got {colors_logits.shape}"
            
            # ^ group the logits as DPL expects: turn each (96,3) logits tensor into a (32,9) logits tensor
            shapes_logits = shapes_logits.view(xs[i].size(0), self.n_predicates)
            colors_logits = colors_logits.view(xs[i].size(0), self.n_predicates)
            assert shapes_logits.shape == (xs[i].size(0), self.n_predicates), f"Expected shapes_logits shape {(xs[i].size(0), 9)}, but got {shapes_logits.shape}"
            assert colors_logits.shape == (xs[i].size(0), self.n_predicates), f"Expected colors_logits shape {(xs[i].size(0), 9)}, but got {colors_logits.shape}"
            
            # ^ concatenate the shapes and colors logits
            lc = torch.cat((shapes_logits, colors_logits), dim=-1)
            assert lc.shape == (xs[i].size(0), 18), f"Expected lc shape {(xs[i].size(0), 18)}, but got {lc.shape}"
            
            pc = self.normalize_concepts(lc)         
            pred, worlds_prob = self.problog_inference(pc)
            cs.append(lc), pCs.append(pc), preds.append(pred)

        clen = len(cs[0].shape)
        
        cs = torch.stack(cs, dim=1) if clen > 1 else torch.cat(cs, dim=1)
        cs = cs - cs.mean() # ^ recenter the logits around 0 as DPL works best in that range
        pCs = torch.stack(pCs, dim=1) if clen > 1 else torch.cat(pCs, dim=1)

        py = self.combine_queries(preds)
        preds = torch.stack(preds, dim=1) if clen > 1 else torch.cat(preds, dim=1)
        
        return {"CS": cs, "YS": py, "pCS": pCs, "PREDS": preds}


    # & Problog inference method (extracts the probability of combined predicates like diffsha and diffcol from raw scores)
    def problog_inference(self, pCs, query=None):
        worlds_tensor = outer_product(
            *torch.split(
                pCs.squeeze(1), 3, dim=-1
            )  
        )  

        worlds_prob = worlds_tensor.reshape(-1, 3**self.n_facts)
        query_prob = torch.zeros(size=(len(pCs), self.n_predicates), device=pCs.device)

        for i in range(self.n_predicates):
            query = i
            query_prob[:, i] = self.compute_query(query, worlds_prob).view(-1)

        return query_prob, worlds_prob
    
    
    # & Combine queries method (computes the probability for chain of predicates defining the formula)
    def combine_queries(self, preds):
        y_worlds = outer_product(*preds).reshape(-1, 9**self.n_images)
        py = torch.zeros(size=(len(preds[0]), self.nr_classes), device=preds[0].device)

        for i in range(self.nr_classes):
            and_rule = self.and_rule[:, i]
            query_prob = torch.sum(and_rule * y_worlds, dim=1, keepdim=True)

            py[:, i] = query_prob.view(-1)

        return py


    # & Computes query probability given the worlds probability P(w).
    def compute_query(self, query, worlds_prob):
        w_q = self.w_q[:, query]
        query_prob = torch.sum(w_q * worlds_prob, dim=1, keepdim=True)

        return query_prob


    # & Normalize concepts method to get pCs for cs
    def normalize_concepts(self, z):
        def soft_clamp(h, dim=-1):
            h = nn.Softmax(dim=dim)(h)
            eps = 1e-5
            h = h + eps
            with torch.no_grad():
                Z = torch.sum(h, dim=dim, keepdim=True)
            h = h / Z
            return h

        pCi = torch.split(z, 3, dim=-1)
        norm_concepts = torch.cat(
            [soft_clamp(c) for c in pCi], dim=-1
        )

        return norm_concepts
    

    @staticmethod
    # & Standard classification loss for this architecture
    def get_loss(args):
        if args.dataset in ["kandinsky", "prekandinsky", "minikandinsky"]:
            return KAND_DPL(KAND_Cumulative)
        else:
            return NotImplementedError("Wrong dataset choice")


    # override of to
    def to(self, device):
        super().to(device)
        self.w_q = self.w_q.to(device)
        self.and_rule = self.and_rule.to(device)