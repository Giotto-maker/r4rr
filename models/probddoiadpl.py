from utils.args import *
from utils.conf import get_device
from utils.losses import SDDOIA_Cumulative
from utils.dpl_loss import SDDOIA_DPL
from models.utils.utils_problog import *
from models.sddoiadpl import SDDOIADPL

# * Parser method 
def get_parser() -> ArgumentParser:
    """Returns the parser

    Returns:
        argparse: argument parser
    """
    parser = ArgumentParser(description="Learning via" "Concept Extractor .")
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class ProBddoiaDPL(SDDOIADPL):
    """Prototypical DPL MODEL FOR BOIA"""

    NAME = "probddoiadpl"
    """
    BOIA
    """

    # & Init method
    def __init__(
        self,
        encoder,
        n_images=1,
        c_split=(),
        args=None,
        model_dict=None,
        n_facts=21,
        nr_classes=4,
    ):
        super(ProBddoiaDPL, self).__init__(
            encoder=encoder,
            model_dict=model_dict,
            args=args,
            n_facts=n_facts,
            nr_classes=nr_classes,
        )

        # ! x-model
        if self.args.expressive_model:
            hidden_dim = getattr(self.args, 'expressive_hidden_dim', self.n_facts * 2)
            self.mlp = nn.Sequential(
                nn.Linear(self.n_facts * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.n_facts * 2)
            )
            self.mlp = self.mlp.to(self.device)


    # & Forward method    
    def forward(self, x, support_emb_dict: dict = None, debug=False):
        assert support_emb_dict is not None, "Support embeddings not provided."
        batch_size = x.shape[0]
        logits_list = []
        raw_logits_list = []
        
        # For each concept, use distances from prototypical predict
        for i in range(self.n_facts):
            support_emb, support_labels = support_emb_dict[i]
            # ^ dists: [batch,2] for classes {absent, present}
            _, _, dists = self.encoder[i].predict(  
                support_emb, support_labels, x
            )
            # ^ normalize distances, still [batch,2]
            p01 = torch.softmax(dists, dim=1)
            logits_list.append(p01)
            if debug:
                print(f"Concept {i}: p0,p1 sample =", p01[0].detach().cpu().numpy())

            raw_logits_list.append(dists)

        # ^ Stack concept logits to have shape [batch, n_facts, 2]
        p01_all = torch.stack(logits_list, dim=1)
        raw_p01_all = torch.stack(raw_logits_list, dim=1)
        cs, pCs, mlp_out_norm = None, None, None
        if not self.args.expressive_model:
            # ^ positive concept scores: [batch, n_facts] 
            cs = p01_all[:,:,1]
            # ^ flat to torch.Size([64, n_facts * 2])
            pCs = p01_all.view(batch_size, self.n_facts * 2)
        # ! x-model
        else:
            # ^ p_raw_flat is [batch, n_facts*2]
            p_raw_flat = raw_p01_all.view(batch_size, self.n_facts * 2)
            mlp_out = self.mlp(p_raw_flat)
            # ^ reshape back to [batch, n_facts, 2] to apply softmax per-concept and get new probabilities
            mlp_out_resh = mlp_out.view(batch_size, self.n_facts, 2)
            mlp_out_norm = torch.softmax(mlp_out_resh, dim=2)
            # ^ positive concept scores: [batch, n_facts] 
            cs = mlp_out_norm[:, :, 1]
            # ^ flat to torch.Size([64, n_facts * 2])
            pCs = mlp_out_norm.view(batch_size, self.n_facts * 2)

        assert cs.shape == (batch_size, self.n_facts)
        assert pCs.shape == (batch_size, self.n_facts * 2)
        for i in range(21):
            # Check if each pair sums to 1
            s = pCs[:, 2 * i] + pCs[:, 2 * i + 1]
            assert torch.allclose(s, torch.ones_like(s), atol=1e-4), f"Softmax pair {i} does not sum to 1: {s}"
            
        # Problog inference
        py = self.problog_inference(pCs)

        if debug:
            print("ProBddoiaDPL forward output:")
            print(f"pCs shape: {pCs.shape}, cs shape: {cs.shape}, py shape: {py.shape}")
            print(f"pCs sample: {pCs[0].detach().cpu().numpy()}")
            print(f"cs sample: {cs[0].detach().cpu().numpy()}")
            print(f"py sample: {py[0].detach().cpu().numpy()}")
            print("pCs (probabilities of concepts):")
            print(pCs)

        return {"CS": cs, "YS": py, "pCS": pCs,
                "pnets_probs": p01_all, "mlp_probs": mlp_out_norm}


    # & Logical inference method with DPL
    def problog_inference(self, pCs, query=None):
        """Performs ProbLog inference to retrieve the worlds probability distribution P(w). Works with two encoded bits.

        Args:
            self: instance
            pCs: probability of concepts
            query (default=None): query

        Returns:
            query_prob: query probability
            worlds_prob: worlds probability
        """

        # for forward
        tl_green = pCs[:, :2]  # traffic light is green
        follow = pCs[:, 2:4]  # follow car ahead
        clear = pCs[:, 4:6]  # road is clear

        # for stop
        tl_red = pCs[:, 6:8]  # traffic light is red
        t_sign = pCs[:, 8:10]  # traffic sign present
        obs = compute_logic_obstacle(self.or_four_bits, pCs)  # generic obstacle

        A = tl_green.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        B = follow.unsqueeze(1).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        C = clear.unsqueeze(1).unsqueeze(2).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        D = tl_red.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(5).unsqueeze(6)
        E = t_sign.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(6)
        F = obs.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)

        w_FS = (
            A.multiply(B).multiply(C).multiply(D).multiply(E).multiply(F).view(-1, 64)
        )
        #
        labels_FS = torch.einsum("bi,ik->bk", w_FS, self.FS_w_q)
        ##

        # for LEFT
        left_lane = pCs[:, 18:20]  # there is LEFT lane
        tl_green_left = pCs[:, 20:22]  # tl green on LEFT
        follow_left = pCs[:, 22:24]  # follow car going LEFT

        # for LEFT-STOP
        no_left_lane = pCs[:, 24:26]  # no lane on LEFT
        l_obs = pCs[:, 26:28]  # LEFT obstacle
        left_line = pCs[:, 28:30]  # solid line on LEFT

        AL = left_lane.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        BL = (
            tl_green_left.unsqueeze(1)
            .unsqueeze(3)
            .unsqueeze(4)
            .unsqueeze(5)
            .unsqueeze(6)
        )
        CL = (
            follow_left.unsqueeze(1).unsqueeze(2).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        )
        DL = (
            no_left_lane.unsqueeze(1)
            .unsqueeze(2)
            .unsqueeze(3)
            .unsqueeze(5)
            .unsqueeze(6)
        )
        EL = l_obs.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(6)
        FL = left_line.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)

        w_L = (
            AL.multiply(BL)
            .multiply(CL)
            .multiply(DL)
            .multiply(EL)
            .multiply(FL)
            .view(-1, 64)
        )

        label_L = torch.einsum("bi,ik->bk", w_L, self.L_w_q)
        ##

        # for RIGHT
        rigt_lane = pCs[:, 30:32]  # there is RIGHT lane
        tl_green_rigt = pCs[:, 32:34]  # tl green on RIGHT
        follow_rigt = pCs[:, 34:36]  # follow car going RIGHT

        # for RIGHT-STOP
        no_rigt_lane = pCs[:, 36:38]  # no lane on RIGHT
        r_obs = pCs[:, 38:40]  # RIGHT obstacle
        rigt_line = pCs[:, 40:42]  # solid line on RIGHT

        AL = rigt_lane.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        BL = (
            tl_green_rigt.unsqueeze(1)
            .unsqueeze(3)
            .unsqueeze(4)
            .unsqueeze(5)
            .unsqueeze(6)
        )
        CL = (
            follow_rigt.unsqueeze(1).unsqueeze(2).unsqueeze(4).unsqueeze(5).unsqueeze(6)
        )
        DL = (
            no_rigt_lane.unsqueeze(1)
            .unsqueeze(2)
            .unsqueeze(3)
            .unsqueeze(5)
            .unsqueeze(6)
        )
        EL = r_obs.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(6)
        FL = rigt_line.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)

        w_R = (
            AL.multiply(BL)
            .multiply(CL)
            .multiply(DL)
            .multiply(EL)
            .multiply(FL)
            .view(-1, 64)
        )

        label_R = torch.einsum("bi,ik->bk", w_R, self.R_w_q)

        pred = torch.cat([labels_FS, label_L, label_R], dim=1)  # this is 8 dim

        # avoid overflow
        pred = (pred + 1e-5) / (1 + 2 * 1e-5)

        return pred


    # & Loss function for the architecture
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
        if args.dataset in ["boia_original_embedded"]:
            return SDDOIA_DPL(SDDOIA_Cumulative)
        else:
            return NotImplementedError("Wrong dataset choice")


    # & creates the JOINT optimizer for the model accounting for all the prototypical backbones
    def start_optim(self, args):
        """Initialize optimizer

        Args:
            self: instance
            args: command line arguments

        Returns:
            None: This function does not return a value.
        """
        # Get all trainable parameters from the encoder modules
        backbones_params = []
        for i in range(21):
            encoder = self.encoder[i]
            # Check if all parameters require gradients
            for name, param in encoder.named_parameters():
                if not param.requires_grad:
                    raise RuntimeError(f"Parameter '{name}' in encoder_{i} does not require gradients.")
            backbones_params += list(encoder.parameters())

        if not backbones_params:
            raise RuntimeError("No trainable parameters found in encoder modules.")

        # Add MLP parameters if expressive model is used
        if self.args.expressive_model:
            for name, param in self.mlp.named_parameters():
                if not param.requires_grad:
                    raise RuntimeError(f"Parameter '{name}' in mlp does not require gradients.")
            backbones_params += list(self.mlp.parameters())

        self.opt = torch.optim.Adam(
            backbones_params, args.lr, weight_decay=args.weight_decay
        )


    # & override default to method to move additional tensors to the device
    def to(self, device):
        super().to(device)
        self.or_four_bits = self.or_four_bits.to(device)

        # Worlds-queries matrix
        if self.args.task == "boia":
            self.FS_w_q = self.FS_w_q.to(device)
            self.L_w_q = self.L_w_q.to(device)
            self.R_w_q = self.R_w_q.to(device)