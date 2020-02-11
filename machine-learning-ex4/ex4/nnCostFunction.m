function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); %25x401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); %10x26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); % 25x401
Theta2_grad = zeros(size(Theta2)); % 10x26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Computing the cost
% =========================================================================
a1 = [ones(m, 1) X]'; % 401x5000

z2 = Theta1 * a1; %25x5000
a2 = sigmoid(z2); %25x5000
a2 = [ones(1,m); a2]; %26x5000

z3 = Theta2 * a2; %10x5000
H = sigmoid(z3); %10x5000

temp = zeros(m, num_labels);
for i=1:m
  temp(i, y(i)) = 1;
end

H = H';

J1 = -1/m * sum(sum(temp.*log(H) + (1-temp).*log(1-H)));

temp1 = Theta1(:, 2:end);
temp2 = Theta2(:, 2:end);
temp1 = temp1.^2;
temp2 = temp2.^2;
J2 = sum(sum(temp1)) + sum(sum(temp2));
J2 = lambda/(2*m)*J2;
J = J1 + J2;


%Computing the gradient
% =========================================================================

for t = 1:m
  %Step 01
  a1 = [1 X(t, :)]'; % 1x401
  z2 = Theta1 * a1; % 25x1
  a2 = sigmoid(z2); % 25x1
  a2 = [1; a2]; % 26x1
  z3 = Theta2 * a2; % 10x1
  a3 = sigmoid(z3); % 10x1
  
  %Step 02
  d3 = a3 - temp(t, :)'; %10x1; temp(t) = vectorized(y(t))
  
  %Step 03
  d2 = (Theta2'*d3).* (a2.*(1-a2)); % 26x1
  
  %Step 04
  Theta2_grad = Theta2_grad + d3*a2'; % 10x26
  Theta1_grad = Theta1_grad + d2(2:end)*a1'; % 25x401

end

%Step 05
Theta1_grad = 1/m*(Theta1_grad + lambda*Theta1);
Theta2_grad = 1/m*(Theta2_grad + lambda*Theta2);

%j = 0
Theta1_grad(:, 1) = Theta1_grad(:, 1) - lambda/m*Theta1(:, 1);
Theta2_grad(:, 1) = Theta2_grad(:, 1) - lambda/m*Theta2(:, 1);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
%fprintf('\nSize of input = %dx%d.', size(X,1), size(X,2));
%fprintf('\nSize of unrolled grad = %dx%d.\n\n', size(grad,1), size(grad, 2));

end
