function [S,D] = transition_matrix(W)

	np = size (W, 1);
	D = full(sum(W,2)).^-0.5;
	D = spdiags (D, 0, np, np);
	S = D * W * D;