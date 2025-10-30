
# Kandinksy DPL
import torch
import torch.nn as nn
from models.utils.deepproblog_modules import DeepProblogModel
from shortcut_mitigation.kandinsky.protonet_kand_modules.yolo_modules.yolo import yolo_detect_and_crop_primitives_batch
from utils.args import *
from utils.conf import get_device
from models.utils.utils_problog import *
from utils.losses import *
from utils.dpl_loss import KAND_DPL
from models.utils.ops import outer_product


def get_parser() -> ArgumentParser:
    """Argument parser for Kandinsky DPL

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class KandDPLSingleJoint(DeepProblogModel):
    """Kandinsky single primitive match with joint encoders DPL model"""

    NAME = "kanddplsinglejoint"
    
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
        super(KandDPLSingleJoint, self).__init__(
            encoder=encoder,
            model_dict=model_dict,
            n_facts=n_facts,
            nr_classes=nr_classes,
        )

        # how many images and explicit split of concepts
        self.n_images = n_images
        self.c_split = c_split

        # Worlds-queries matrix
        # if args.task == 'base':
        self.n_facts = 6
        self.w_q, self.and_rule = build_worlds_queries_matrix_KAND(
            self.n_images, self.n_facts, 3, task=args.task
        )
        self.n_predicates = 9
        self.nr_classes = 2

        # opt and device
        self.opt = None
        self.device = get_device()
        self.w_q = self.w_q.to(self.device)
        self.and_rule = self.and_rule.to(self.device)

        
    # & Forward function
    def forward(self, x, concept_extractor, transform, args):
        """
        Forward pass for processing a batch of images and extracting concept representations.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels * n_images, height, width),
                where n_images is the number of images per sample.
            concept_extractor (callable): Function or model used to extract concepts from images.
            transform (callable): Transformation function to apply to images before concept extraction.
            args (Namespace or dict): Additional arguments required for concept extraction and processing.
        Returns:
            dict: A dictionary containing:
                - "CS" (torch.Tensor): Raw concept representations for each image in the batch.
                - "YS" (torch.Tensor): Combined predictions for the batch.
                - "pCS" (torch.Tensor): Normalized concept representations.
                - "PREDS" (torch.Tensor): Individual predictions for each image in the batch.
        Raises:
            AssertionError: If the input shapes do not match the expected dimensions at various steps.
        """

        cs, pCs, preds = [], [], []
        xs = torch.split(x, x.size(-1) // self.n_images, dim=-1)
        for i in range(self.n_images):
            assert xs[i].shape == (x.size(0), 3, 64, 64), f"Expected input shape {(x.size(0), 3, 64, 64)}, but got {xs[i].shape}"
            cropped_objects = yolo_detect_and_crop_primitives_batch(xs[i], concept_extractor, transform, args)
            lc = self.encoder(cropped_objects)
            lc = lc.view(xs[i].size(0), self.n_predicates*2)
            assert lc.shape == (xs[i].size(0), 18), f"Expected lc shape {(xs[i].size(0), 18)}, but got {lc.shape}"
            pc = self.normalize_concepts(lc)         
            pred, worlds_prob = self.problog_inference(pc)
            cs.append(lc), pCs.append(pc), preds.append(pred)

        clen = len(cs[0].shape)        
        cs = torch.stack(cs, dim=1) if clen > 1 else torch.cat(cs, dim=1)
        cs = cs - cs.mean()
        pCs = torch.stack(pCs, dim=1) if clen > 1 else torch.cat(pCs, dim=1)
        py = self.combine_queries(preds)
        preds = torch.stack(preds, dim=1) if clen > 1 else torch.cat(preds, dim=1)

        return {"CS": cs, "YS": py, "pCS": pCs, "PREDS": preds}
    

    def problog_inference(self, pCs, query=None):
        """Problog inference

        Args:
            self: instance
            pCs: probabilities of concepts
            preds: predictions

        Returns:
            query_prob: query probabilities
            worlds_prob: worlds probabilities
        """

        # Extract probs of shapes and colors for each object
        worlds_tensor = outer_product(
            *torch.split(
                pCs.squeeze(1), 3, dim=-1
            )  # [batch_size, 1, 3*8] -> [8, batch_size, 3]
        )  # [8, batch_size, 3] -> [batch_size, 3,3,3,3, 3,3,3,3]

        worlds_prob = worlds_tensor.reshape(-1, 3**self.n_facts)

        # Compute query probability P(q)
        query_prob = torch.zeros(size=(len(pCs), self.n_predicates), device=pCs.device)

        for i in range(self.n_predicates):
            query = i
            query_prob[:, i] = self.compute_query(query, worlds_prob).view(-1)

        # add a small offset
        # query_prob += 1e-5
        # with torch.no_grad():
        #     Z = torch.sum(query_prob, dim=-1, keepdim=True)
        # query_prob = query_prob / Z

        return query_prob, worlds_prob

    def combine_queries(self, preds):
        """Combine queries

        Args:
            self: instance
            preds: predictions

        Returns:
            py: pattern probabilities
        """
        y_worlds = outer_product(*preds).reshape(-1, 9**self.n_images)

        py = torch.zeros(size=(len(preds[0]), self.nr_classes), device=preds[0].device)

        for i in range(self.nr_classes):
            and_rule = self.and_rule[:, i]
            query_prob = torch.sum(and_rule * y_worlds, dim=1, keepdim=True)

            py[:, i] = query_prob.view(-1)

        return py

    def compute_query(self, query, worlds_prob):
        """Computes query probability given the worlds probability P(w).

        Args:
            self: instance
            query: query
            worlds_prob: world probabilities

        Returns:
            query_prob: query probabilities
        """
        # Select the column of w_q matrix corresponding to the current query
        w_q = self.w_q[:, query]
        # Compute query probability by summing the probability of all the worlds where the query is true
        query_prob = torch.sum(w_q * worlds_prob, dim=1, keepdim=True)

        # for concepts in torch.split(latents):
        #     s1, c1, s2, c2, s3, c3, s4, c4 = torch.split(concepts, 3, dim=-1)
        return query_prob

    def normalize_concepts(self, z):
        """Computes the probability for each ProbLog fact given the latent vector z

        Args:
            self: instance
            z (torch.tensor): latents

        Returns:
            norm_concepts: normalized concepts
        """

        def soft_clamp(h, dim=-1):
            h = nn.Softmax(dim=dim)(h)

            eps = 1e-5
            h = h + eps
            with torch.no_grad():
                Z = torch.sum(h, dim=dim, keepdim=True)
            h = h / Z
            return h

        # TODO: the 3 here is hardcoded, relax to arbitrary concept encodings?
        pCi = torch.split(z, 3, dim=-1)  # [batch_size, 24] -> [8, batch_size, 3]

        norm_concepts = torch.cat(
            [soft_clamp(c) for c in pCi], dim=-1
        )  # [8, batch_size, 3] -> [batch_size, 24]

        return norm_concepts

    @staticmethod
    def get_loss(args):
        """Loss function for KandDPL

        Args:
            args: command line arguments

        Returns:
            loss: loss function

        Raises:
            err: NotImplementedError
        """
        if args.dataset in ["kandinsky", "prekandinsky", "minikandinsky"]:
            return KAND_DPL(KAND_Cumulative)
        else:
            return NotImplementedError("Wrong dataset choice")

    def start_optim(self, args):
        """Initialize optimizer

        Args:
            args: command line arguments

        Returns:
            None: This function does not return a value.
        """
        self.opt = torch.optim.Adam(
            self.parameters(),
            args.lr,
            weight_decay=1e-5,
        )

    # override of to
    def to(self, device):
        super().to(device)
        self.w_q = self.w_q.to(device)
        self.and_rule = self.and_rule.to(device)