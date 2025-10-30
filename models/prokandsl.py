import torch
import torch.nn as nn

from utils.args import *
from utils.conf import get_device
from utils.losses import *
from utils.semantic_loss import KANDINSKY_SL

from models.cext import CExt
from models.utils.utils_problog import build_worlds_queries_matrix_KAND
from models.utils.ops import outer_product

from shortcut_mitigation.kandinsky.protonet_kand_modules.utility_modules.visualizers import plot_primitives
from shortcut_mitigation.kandinsky.protonet_kand_modules.yolo_modules.yolo import yolo_detect_and_crop_primitives_batch


# * Parser method
def get_parser() -> ArgumentParser:
    """Returns the argument parser for this architecture"""
    parser = ArgumentParser(description="Learning via Kandinsky Concept Extractor.")
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
class ProKandSL(CExt):
    """Kandinsky DPL model"""
    NAME = "prokandsl"

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
        super(ProKandSL, self).__init__(encoder=encoder, n_images=n_images, c_split=c_split)

        # ^ register the encoders as a parameter
        self.encoder = nn.ModuleList(encoder)

        # Logical world-query matrix
        self.n_facts = n_facts
        self.logic, self.and_rule = build_worlds_queries_matrix_KAND(
            self.n_images, self.n_facts, 3, task=args.task
        )
        self.nr_classes = nr_classes
        self.n_predicates = 9

        assert self.n_images == 3, "KandinskySL expects 3 subimages"
        assert self.n_facts == 6, "KandinskySL expects 3 shapes × 2 colors"
        assert self.nr_classes == 2, "KandinskySL only supports binary classification"
        assert self.logic.shape == (729, self.n_predicates), f"Expected shape (729, {self.nr_classes}), but got {self.logic.shape}"
        
        # Optimizer and device configuration
        self.opt = None
        self.device = get_device()
        self.logic = self.logic.to(self.device)
        self.and_rule = self.and_rule.to(self.device)

        

    # & Forward method
    def forward(self, x, concept_extractor, transform, support_images, support_labels, args, unknown_init=False):    
        """
        Forward pass for the model, processing input images and support data to extract and normalize concept logits,
        compute predicate probabilities, and aggregate predictions.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width * n_images), containing concatenated images.
            concept_extractor (callable): Function or model used to extract concepts from images.
            transform (callable): Transformation function to apply to images before concept extraction.
            support_images (torch.Tensor): Support images tensor used for few-shot learning or concept matching.
            support_labels (torch.Tensor): Support labels tensor, typically of shape (num_support, 2) for shape and color labels.
            args (Namespace): Arguments namespace containing configuration options such as debug flags and hidden classes.
            unknown_init (bool, optional): Flag indicating whether to initialize unknown concepts. Defaults to False.
        Returns:
            dict: A dictionary containing the following keys:
                - "CS": Raw concatenated concept logits for each image in the batch.
                - "YS": Soft-inferred predictions for each image in the batch.
                - "pCS": Normalized concept logits for each image in the batch.
                - "preds": Predicate predictions for each image in the batch.
                - "gpreds": Aggregated predicate and label probabilities for the batch.
        """
        
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
            
            _, _, shapes_logits = self.encoder[0].predict(support_images_shapes, support_shape_labels, cropped_objects, unknown_init=unknown_init) # ^ (96,3)
            assert shapes_logits.shape == (xs[i].size(0)*3, 3), f"Expected shapes_logits shape {(xs[i].size(0), 3)}, but got {shapes_logits.shape}"
            _, _, colors_logits = self.encoder[1].predict(support_images_colors, support_color_labels, cropped_objects, unknown_init=unknown_init) # ^ (96,3)
            assert colors_logits.shape == (xs[i].size(0)*3, 3), f"Expected colors_logits shape {(xs[i].size(0), 3)}, but got {colors_logits.shape}"
            
            # ^ group the logits as SL expects: turn each (96,3) logits tensor into a (32,9) logits tensor
            shapes_logits = shapes_logits.view(xs[i].size(0), self.n_predicates)
            colors_logits = colors_logits.view(xs[i].size(0), self.n_predicates)
            assert shapes_logits.shape == (xs[i].size(0), self.n_predicates), f"Expected shapes_logits shape {(xs[i].size(0), 9)}, but got {shapes_logits.shape}"
            assert colors_logits.shape == (xs[i].size(0), self.n_predicates), f"Expected colors_logits shape {(xs[i].size(0), 9)}, but got {colors_logits.shape}"
            
            # ^ concatenate the shapes and colors logits
            lc = torch.cat((shapes_logits, colors_logits), dim=-1)
            assert lc.shape == (xs[i].size(0), 18), f"Expected lc shape {(xs[i].size(0), 18)}, but got {lc.shape}"
            
            pc = self.normalize_concepts(lc)         
            pred, worlds_prob = self.compute_combined_predicates_probability(pc)
            cs.append(lc), pCs.append(pc), preds.append(pred)

            unknown_init = False

        clen = len(cs[0].shape)
        
        cs = torch.stack(cs, dim=1) if clen > 1 else torch.cat(cs, dim=1)
        pCs = torch.stack(pCs, dim=1) if clen > 1 else torch.cat(pCs, dim=1)
        
        py = self.soft_inference(preds)
        
        preds = torch.stack(preds, dim=1) if clen > 1 else torch.cat(preds, dim=1)
        aggregated_preds_and_label = self.compute_predicates_label_probability(preds, py, x.size(0))

        return {"CS": cs, "YS": py, "pCS": pCs, "preds": preds, "gpreds": aggregated_preds_and_label}


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


    # & Symbolic reasoner to get the predictions from the atomic concepts predictions
    '''
        # Query Index Mapping:
        # 
        # | Query Index | Predicate           | Shape Rule     | Color Rule     |
        # |-------------|---------------------|----------------|----------------|
        # | 0           | diffsha & diffcol   | All different  | All different  |
        # | 1           | diffsha & twocol    | All different  | Two-of-a-kind  |
        # | 2           | diffsha & samecol   | All different  | All same       |
        # | 3           | twosha  & diffcol   | Two-of-a-kind  | All different  |
        # | 4           | twosha  & twocol    | Two-of-a-kind  | Two-of-a-kind  |
        # | 5           | twosha  & samecol   | Two-of-a-kind  | All same       |
        # | 6           | samesha & diffcol   | All same       | All different  |
        # | 7           | samesha & twocol    | All same       | Two-of-a-kind  |
        # | 8           | samesha & samecol   | All same       | All same       |
    '''
    def compute_combined_predicates_probability(self, pCs, query=None):
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
    

    # & Combine queries method
    def soft_inference(self, preds):
        y_worlds = outer_product(*preds).reshape(-1, 9**self.n_images)

        py = torch.zeros(size=(len(preds[0]), self.nr_classes), device=preds[0].device)

        for i in range(self.nr_classes):
            and_rule = self.and_rule[:, i]
            query_prob = torch.sum(and_rule * y_worlds, dim=1, keepdim=True)
            py[:, i] = query_prob.view(-1)

        return py


    # & Computes query probability given the worlds probability P(w).
    # Select the column of logic matrix corresponding to the current query and 
    # compute query probability by summing the probability of all the worlds where the query is true
    def compute_query(self, query, worlds_prob):
        logic = self.logic[:, query]
        query_prob = torch.sum(logic * worlds_prob, dim=1, keepdim=True)

        return query_prob
    

    # & Given the combined probs from compute_combined_predicates_probability, get probs for each predicate
    def compute_predicates_label_probability(self, high_concepts, label, batch_size):
        aggregated_preds = []  
        for pred in high_concepts:  # each pred is (batch, 9)
            p_diff_shape = pred[:, 0] + pred[:, 1] + pred[:, 2]
            p_twoshape   = pred[:, 3] + pred[:, 4] + pred[:, 5]
            p_same_shape = pred[:, 6] + pred[:, 7] + pred[:, 8]
            
            p_diff_color = pred[:, 0] + pred[:, 3] + pred[:, 6]
            p_twocolor   = pred[:, 1] + pred[:, 4] + pred[:, 7]
            p_same_color = pred[:, 2] + pred[:, 5] + pred[:, 8]
            
            image_pred = torch.stack([p_diff_shape, p_twoshape, p_same_shape,
                                      p_diff_color, p_twocolor, p_same_color], dim=1)  # (batch, 6)
            aggregated_preds.append(image_pred)
        
        # Now, 'aggregated_preds' is a list of 3 tensors, each of shape (batch,6)
        images_tensor = torch.stack(aggregated_preds, dim=1).permute(1, 0, 2)
        assert images_tensor.shape == (batch_size, 3, 6), \
            f"Expected images_tensor shape {batch_size, 3, 6}, but got {images_tensor.shape}"
        
        # Pad the final label with 0s to match the shape of the constrained variables
        label_tensor = torch.cat([label, torch.zeros(label.shape[0], 4, device=label.device)], dim=1)  # (batch,6)
        
        assert images_tensor.shape == (batch_size, 3, 6), \
            f"Expected images_tensor shape {batch_size, 3, 6}, but got {images_tensor.shape}"
        assert label_tensor.unsqueeze(1).shape == (batch_size, 1, 6), \
            f"Expected label_tensor shape {batch_size, 1, 6}, but got {label_tensor.unsqueeze(1).shape}"
        
        final_tensor = torch.cat([images_tensor, label_tensor.unsqueeze(1)], dim=1)  # (batch, 4, 6)
        return final_tensor

    
    # & Semantic loss method
    def get_loss(self, args):
        if args.dataset in ["kandinsky"]:
            return KANDINSKY_SL(KAND_Cumulative, args)
        else:
            raise NotImplementedError("Dataset not supported for KandinskySL")


    # override of to
    def to(self, device):
        super().to(device)
        self.logic = self.logic.to(device)
        self.and_rule = self.and_rule.to(device)
