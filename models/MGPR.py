import gpflow


class MGPR(gpflow.Parameterized):
    """
    We maintain multiple Gaussian Process for every dimension of the output.
    """
    def __init__(self, X, Y):
        super(MGPR, self).__init__()

        self.output_dim = Y.shape[1]
        self.input_dim = X.shape[1]
        self.num_data = X.shape[0]

        self.gps = []
        self.create_gps(X, Y)

    def create_gps(self, X, Y):
        self.gps.clear()

        for i in range(self.output_dim):
            kern = gpflow.kernels.RBF(input_dim=X.shape[1], ARD=True)
            self.models.append(gpflow.models.GPR(X, Y[:, i:i+1], kern))
            self.models[i].clear()
            self.models[i].compile()

    def set_XY(self, X, Y):
        for i in range(len(self.models)):
            self.gps[i].X = X
            self.gps[i].Y = Y[:, i:i+1]

    def optimize(self):
        optimizer = gpflow.train.ScipyOptimizer(options={'maxfun': 500})
        for model in self.models:
            optimizer.minimize(model)






