function y = first(x)

if iscell(x), y = x{1};
else          y = x;
end
