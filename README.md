## Deep Galerkin Method for Pedestrian MFG

When considering the system of equations of the pedestrian MFG with both discount factor and congestion:
$$\frac{\sigma^2}{2}\Delta{u} - \dfrac{(\vec{\nabla}u)^2}{2\mu (1+\alpha m )} -\lambda -\vec{s}\cdot\vec{\nabla}u -  \gamma u =  V[m] \qquad \text{(HJB)}$$
 $$\frac{\sigma^2}{2}\Delta m + \vec{s}\cdot\nabla m +\dfrac{1}{\mu}\nabla\cdot\left(\dfrac{m}{1+\alpha m}\vec{\nabla}u\right) =0 \qquad \;\text{(FP)}$$
 we aim at retrieving the solution by means of the minimization of the residuals 
    $$\underset{u,m}{argmin}  \{J (u,m) := a\big\|r_{HJB}(u,m)\big\|^2_{2,\Omega} + b\big\|r_{FP}(u,m)\big\|^2_2+c_1\|m\big\|^2_{2,\delta C} + c_2\|m - m_B\big\|^2_{2,\delta R} + d\|u-u_B\big\|^2_{2,\delta R}\}$$
where $\delta R$, $\delta C $ are the boundary of room and cylinder respectively, $\Omega$ collects all the point inside the room (cylinder excluded), and
    $$r_{HJB}(u,m) = \frac{\sigma^2}{2}\Delta{u} - \dfrac{(\vec{\nabla}u)^2}{2\mu (1+\alpha m )} -\lambda -\vec{s}\cdot\vec{\nabla}u -  \gamma u -  V[m]$$
    $$r_{FP}(u,m) = \frac{\sigma^2}{2}\Delta m + \vec{s}\cdot\nabla m +\dfrac{1}{\mu}\nabla\cdot\left(\dfrac{m}{1+\alpha m}\vec{\nabla}u\right)\,$$
    $$m_B = m_0 $$
    $$u_B = -\mu\sigma^2 log(m_0)$$
We consider a cylinder of radius $R=0.37$, and the initial condition for the mass of pedestrians $m_0 = 2.5$. This constant enters as the room boundary condition for $m$, since we assume that far away from the cylinder, the pedestrians are unaffected. A quantification of how much it is *far enough* can be computed by means of the *healing length*
    $$\xi = \sqrt{\dfrac{\mu\sigma^4}{|2 g m_0|}}$$
    
   
