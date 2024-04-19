import numpy as np

from f1tenth_gym.envs.laser_models import ScanSimulator2D


def main():
    num_test = 18
    num_beams = 100
    fov = 4.7
    map_name = "Spielberg"
    scan_rng = np.random.default_rng(seed=0)
    scan_sim = ScanSimulator2D(num_beams, fov)
    scan_sim.set_map(map_name)
    new_scan = np.empty((num_test, num_beams))

    test_poses = np.zeros((num_test, 3))
    test_poses[:, 2] = np.linspace(-1.0, 1.0, num=num_test)

    # scan gen loop
    for i in range(num_test):
        test_pose = test_poses[i]
        new_scan[i, :] = scan_sim.scan(pose=test_pose, rng=scan_rng)
    # diff = self.sample_scans[map_name] - new_scan
    # mse = np.mean(diff ** 2)

    # plotting
    import matplotlib.pyplot as plt

    theta = np.linspace(-fov / 2.0, fov / 2.0, num=num_beams)

    # plar表示
    for i in range(num_test):
        plt.polar(theta, new_scan[i, :], ".", lw=0)
        plt.show()

    # plt.polar(theta, new_scan[1, :], ".", lw=0)

    # 点の角度ごとに色を変えてプロット
    plt.scatter(theta, new_scan[1, :], c=theta, cmap="hsv")
    plt.xlabel("theta")
    plt.ylabel("scan dist")
    plt.show()

    # 点をsparseにする
    num_sparse_points = 10
    sparse_idx = np.linspace(0, num_beams - 1, num=num_sparse_points, dtype=int)
    plt.scatter(
        theta[sparse_idx], new_scan[1, sparse_idx], c=theta[sparse_idx], cmap="hsv"
    )

    plt.xlabel("theta")
    plt.ylabel("scan dist")

    plt.show()


if __name__ == "__main__":
    main()
