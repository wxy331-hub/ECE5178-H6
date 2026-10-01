import torch

# Fixed feature scales.  They live here, not in a file beside the weights, so
# weight.pth loads into this class exactly as the submission requires.
POSITION_SCALE = 0.625  # half the maze's width, m
SPEED_SCALE = 0.20      # about the robot's cruising speed, m/s
CRUISE_SPEED = 0.15     # the only speed command Lab 3 drove the robot with
MAX_TURN_RATE = 7.5     # rad/s, the calibrated model's limit


class Policy(torch.nn.Module):
    """
    Neural network policy for the Sphero robot, trained by behaviour cloning.

    SUBMISSION REQUIREMENTS:
      - Upload your trained weights and name the file `weight.pth` with this file.
      - The weights must be loadable with:
            policy.load_state_dict(torch.load("./weight.pth"))
        so save them with torch.save(policy.state_dict(), "./weight.pth").
      - The model must be callable as:
            action = policy(torch.tensor(obs, dtype=torch.float32)).detach().numpy()
        where:
            obs    : the observed state, in the format [x, y, heading, speed]
            action : the returned action, in the format [speed, turn_rate_cmd]

    The teacher only ever commands two speeds -- stopped, to turn on the spot,
    or cruising -- so the first output is a drive/stop decision rather than a
    regressed speed that would hover near the motor deadband.  The heading
    enters as its sine and cosine, so +pi and -pi are the same input.
    ``turn_rate_cmd`` is in rad/s.
    """

    def __init__(self, d_in=4, hidden=128, d_out=2):
        super(Policy, self).__init__()
        if d_in != 4 or d_out != 2:
            raise ValueError("the policy maps [x, y, heading, speed] to [speed, turn_rate_cmd]")
        self.model = torch.nn.Sequential(
            torch.nn.Linear(5, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, d_out),
        )

    @staticmethod
    def features(obs):
        x, y, heading, speed = obs.unbind(-1)
        return torch.stack(
            [x / POSITION_SCALE, y / POSITION_SCALE,
             torch.sin(heading), torch.cos(heading), speed / SPEED_SCALE],
            dim=-1,
        )

    def raw_outputs(self, obs):
        """[drive logit, unsquashed turn rate]: what training fits."""
        return self.model(self.features(obs))

    def forward(self, x):
        raw = self.raw_outputs(x)
        drive = raw[..., 0] > 0.0
        speed = torch.where(drive, torch.full_like(raw[..., 0], CRUISE_SPEED),
                            torch.zeros_like(raw[..., 0]))
        rate = MAX_TURN_RATE * torch.tanh(raw[..., 1])
        return torch.stack([speed, rate], dim=-1)
