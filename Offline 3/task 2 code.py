def plot_gaussian(self, data, means, covariances, K, responsibilities):
    
        plt.scatter(data[:, 0], data[:, 1], c=responsibilities.argmax(axis=1), cmap='viridis', s=40, edgecolor='k',
                    alpha=0.2, marker='.')
        x, y = np.mgrid[np.min(data[:, 0]):np.max(data[:, 0]):.01, np.min(data[:, 1]):np.max(data[:, 1]):.01]
        positions = np.dstack((x, y))
        for j in range(K):
            rv = mvn(means[j], covariances[j])
            plt.contour(x, y, rv.pdf(positions), colors='black', alpha=0.6, linewidths=1)

    def animate(self, data, iterations = 100, n_components = 3):

        K = n_components
        n_samples, n_features = data.shape

        if n_features != 2:
            print("Drawing animation is only supported for 2D data")
            return

        # Initializing the weights
        weights = np.ones(n_components) / n_components
        # Initializing the mean vector
        means = np.random.rand(self.n_components,n_features)
        # Initializing the covariance matrix
        covariances = [np.eye(n_features) for _ in range(n_components)]

        # Create the animation
        fig = plt.figure()
        plt.ion()

        for i in range(iterations):
            # Run the E-step
            responsibilities = self._e_step(data, means, covariances, weights,n_samples, n_components)
            # Run the M-step
            weights, means, covariances = self._m_step(data, means, covariances, weights, responsibilities, n_samples,n_features, n_components)
            # Compute the log-likelihood
            # log_likelihoods.append(log_likelihood(data, weights, means, covariances))
            # Plot the updated Gaussian distributions
            plt.clf()
            self.plot_gaussian(data, means, covariances, K, responsibilities)
            plt.title("Iteration {}".format(i))
            plt.pause(0.005)

        plt.ioff()