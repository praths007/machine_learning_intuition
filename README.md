# machine_learning_intuitions

## gradient descent intuition

   gradient = theta - d (cost)/ d theta   
 (the derivative i.e. change in cost wrt theta tells at what dtheta (smallest change in 
 weight/ theta) the change in cost is MAXIMUM. if each partial derivate is a scalar
 the answer is a simple scalar quantity, 
 but if they are vectors (dot product of 2 vectors
 return a scalar) then if the point is in same direction then quantity is maximized
 this shows that gradient always points in direction of steepest ascent.<br/>
 |a|.|b| = ab cos(theta) [dot product of 2 vectors]<br/>
 quantity is maximum when cos(theta) = 1, theta = 0 <br/>
 
why gradient points in the direction of steepest ascent <br/>
https://math.stackexchange.com/questions/223252/why-is-gradient-the-direction-of-steepest-ascent <br/>
we go in the reverse direction for descent <br/>

similarly, the partial derivative of a function f() is the slope of the line 
tangent to the curve made by function f() at a paritcular point.
now the increase in cost will be maximum if the next point is in the direction of 
increasing slope, and if we go in reverse direction the cost will decrease by maximum
