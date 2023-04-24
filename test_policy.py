# compute average reward per test episode with trained policy


from environment import Environment


def test_policy(env):
    test_rew = 0.  # initialize reward tracking

    for i in range(100):  # loop over 100 test episodes
        obs = env.reset('testing')  # get initial obs

        for j in range(200):  # loop over 200 steps per episode
            act = ...  # TODO: get action for the obs from your trained policy
            rew, next_obs, _ = env.step(act)  # take one step in the environment
            test_rew += rew  # track rewards
            obs = next_obs  # continue from the new obs

    avg_test_rew = test_rew / 100  # compute the average reward per episode

    print(avg_test_rew)  # print the result


if __name__ == '__main__':

    data_dir = ...  # TODO: specify relative path to data directory (e.g., './data', not './data/variant_0')
    variant = ...  # TODO: specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)
    env = Environment(variant=variant, data_dir=data_dir)  # initialize the environment

    test_policy(env)  # test the trained policy
