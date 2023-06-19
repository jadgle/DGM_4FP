# Deep Galerkin Method for MFG

Following the work of [Bonnemain et al.](https://arxiv.org/abs/2201.08592), it appears that Mean-Field Games models do a good job at describing long-term anticipation patterns in optimizations problems involving a large number of competing agents. However, there are times where a forecasting is reasonable only for a portion of the future. In [Butano et al.](https://arxiv.org/abs/2302.08945), it was shown that the aforementioned MFG model can be adapted to such instances, simply by introducing a discount factor tuning the agents' foresightedness. When dealing with a cylindrical intruder crossing a static dense crowd, it has been shown that the experimental data gathered in [Nicolas et al.](https://arxiv.org/abs/1810.03343) can be well reproduced by the stationary state of a MFG whose equations in the Schrödinger formulations are

$$\lambda \Phi^e - \mu\sigma^2\vec{s}\cdot\vec{\nabla}\Phi^e =- \frac{\mu \sigma^4}{2}\Delta \Phi^e -V[m^e]\Phi^e+\gamma\mu\sigma^2\Phi^e\log{\Phi^e},$$

$$-\lambda \Gamma^e -  \mu \sigma^2 \vec{s}\cdot \vec{\nabla}\Gamma^e= \frac{\mu\sigma^4}{2} \Delta \Gamma^e + V[m^e]\Gamma^e -\gamma \mu \sigma^2\Gamma^e \log{\Phi^e}.$$

The present repository contains the code to solve these two equations using Neural Networks, in particular by including relevant physical features of the system, something called Physics Informed Neural Networks (PINNs). In particular, we take inspiration from the work of [Sirignano and Spiliopoulos](https://arxiv.org/abs/1708.07469) and reproduce $\Phi$ and $\Gamma$ with $\Phi_{\theta}$ and $\Gamma_{\theta}$, two NNs composed of an equal number of LSTM and FNN layers. We want these nets to be a solution for the equations above and we do so by making them minimize 

$$\text{loss} = MSE(res_{\text{HJB}}) +  MSE(res_{\text{KFP}}) + MSE(boundary) + MSE(obstacle) + MSE(mass)$$

where the first two terms ensure that the nets solve the equations, more precisely, 

$$res_{\text{HJB}} = -\lambda \Phi_{\theta}- \frac{\mu \sigma^4}{2}\Delta \Phi_{\theta} -V[m_{\theta}]\Phi_{\theta} + \mu\sigma^2\vec{s}\cdot\vec{\nabla}\Phi_{\theta} +\gamma\mu\sigma^2\Phi_{\theta}\log{\Phi_{\theta}},$$

$$res_{\text{KFP}} =\lambda \Gamma_{\theta} + \frac{\mu\sigma^4}{2} \Delta \Gamma_{\theta} + V[m_{\theta}]\Gamma_{\theta} + \mu \sigma^2 \vec{s}\cdot \vec{\nabla}\Gamma_{\theta}-\gamma \mu \sigma^2\Gamma_{\theta} \log{\Phi_{\theta}},$$

the boundary one checks that far from the obstacle the density of pedestrians is at rest at a given value, the obstacle term makes sure no one is under the cylinder and the mass term prevent agents from disappearing.
