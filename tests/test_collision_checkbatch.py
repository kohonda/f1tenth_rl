import torch


def collision_check_torch(
    scans: torch.Tensor,
    vels: torch.Tensor,
    cosines: torch.Tensor,
    side_distances: torch.Tensor,
    ttc_threshold: float,
) -> torch.Tensor:
    """
    Check the iTTC for each scan and return the collision flag for each scan in the batch.

    Args:
        scans: Tensor of shape (batch_size, num_beans) representing the distances measured by each beam in each scan.
        vels: Tensor of shape (batch_size,) representing the velocities for each scan in the batch.
        cosines: Tensor of shape (num_beams,) containing the precomputed cosines of the scan angles.
        side_distances: Tensor of shape (num_beans,) containing the precomputed distances from the laser to the sides of the car for each beam.
        ttc_threshold: float representing the threshold below which a collision is considered imminent.

    Returns:
        torch.Tensor: A boolean tensor of shape (batch_size,) indicating whether a collision is imminent for each scan.
    """
    # Prevent division by zero by replacing zero velocities with a very small value (avoiding in-place operations to maintain autograd)
    vels = torch.where(
        vels == 0, torch.tensor(1e-10, device=vels.device, dtype=vels.dtype), vels
    )

    # Calculate projected velocities for each beam in all scans
    proj_vels = vels.unsqueeze(1) * cosines.unsqueeze(
        0
    )  # Shape: (batch_size, num_beams)

    # Calculate time-to-collision for each beam in all scans
    ttcs = (
        scans - side_distances.unsqueeze(0)
    ) / proj_vels  # Shape: (batch_size, num_beams)

    # Identify beams with TTC within the specified threshold
    collisions = (ttcs < ttc_threshold) & (ttcs >= 0)  # Shape: (batch_size, num_beams)

    # Check if there is any beam within the collision threshold for each scan
    in_collisions = collisions.any(dim=1)  # Shape: (batch_size,)

    return in_collisions


# Test the PyTorch implementation
batch_size = 3
num_beams = 5
ttc_threshold = 4.0

# Example inputs
scans = torch.tensor(
    [
        [10.0, 5.0, 15.0, 20.0, 3.0],
        [12.0, 4.0, 16.0, 21.0, 2.0],
        [11.0, 6.0, 14.0, 19.0, 4.0],
    ]
)

vels = torch.tensor([1.0, 2.0, 0.0])  # Including a zero velocity to test the handling

cosines = torch.tensor([1.0, 0.5, 0.5, 1.0, 0.5])

side_distances = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])

# Run the collision check
is_collisions = collision_check_torch(
    scans, vels, cosines, side_distances, ttc_threshold
)

print(is_collisions)
