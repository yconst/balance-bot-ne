import balance_bot_ne

EVAL = 6
env = balance_bot_ne.BalancebotEnvUneven(render=False)
nn_shape = [env.observation_space_size(), 24, env.action_space_size()]
nn = balance_bot_ne.NeuralNet(nn_shape)

def run_episode(env, policy):
    total_reward = 0
    for i in range(EVAL):
        obs = env.reset()
        done = False
        
        while not done:
            obs, reward, done = env.step(policy.forward(obs)[0])
            total_reward += reward

    return total_reward * (1.0/float(EVAL))

def evaluate(individual):
    nn.adopt_parameters(individual)
    return run_episode(env, nn),

def main():
    ea = balance_bot_ne.EvolAlgorithm(objective_function=evaluate,
                                      param_size=nn.parameter_size(),
                                      pop_size=200)
    ea.run(50)
    best = ea.best

    print("Saving best individual with fitness {0:.5f}".format(best.fitness.values[0]))
    with open("best.pkl", 'wb') as handler:
        pickle.dump({"parameters":list(best), "shape":nn_shape}, handler) 

if __name__ == "__main__":
    main()
    