
function [result,H,obj_o,alpha,beta] = BPSA(X,classnum,knn,flag1,flag2) 

NITER = 20;
viewnum = length(X);   
[num,~] = size(X{1}); 
d = zeros(viewnum,1);
for i = 1:viewnum
    d(i) = size(X{i},2);
end

%% ===================== Preprocessing =====================
for i = 1:viewnum
    for  j = 1:num
         X{i}(j,:) = ( X{i}(j,:) - mean( X{i}(j,:) ) ) / std( X{i}(j,:) );
    end
end

%% ===================== Initialization =====================
% initialize Uv
S = cell(viewnum,1);
Av_rep = zeros(num);
U = cell(viewnum,1);
for v = 1:viewnum
    S{v} = constructW_PKN(X{v}',knn);
    Av_rep = Av_rep+S{v};   
    Lv = Ls(S{v});
    eigvec = eig1(full(Lv),classnum+1,0);
    if flag1 == 1
       U{v} = eigvec(:,2:classnum+1);
    elseif flag1 == 2
       U{v} = eigvec(:,2:classnum+1);
       U{v} = U{v}./repmat(sqrt(sum(U{v}.^2,2)),1,classnum); 
       U{v} = orth(U{v});
    elseif flag1 == 3
       U{v} = eigvec(:,1:classnum);
    elseif flag1 == 4
       U{v} = eigvec(:,1:classnum);
       U{v} = U{v}./repmat(sqrt(sum(U{v}.^2,2)),1,classnum);
       U{v} = orth(U{v});
    end
end

% initialize H
Av_rep = 1/viewnum*Av_rep;
L_rep = Ls(Av_rep);
Y_rep = eig1(L_rep,classnum+1,0);
if flag2 == 1
   H = Y_rep(:,2:classnum+1); 
elseif flag2 == 2
   H = Y_rep(:,2:classnum+1);            
   H = H./repmat(sqrt(sum(H.^2,2)),1,classnum); 
   H = orth(H);
elseif flag2 == 3
   H = Y_rep(:,1:classnum);    
elseif flag2 == 4
   H = Y_rep(:,1:classnum);            
   H = H./repmat(sqrt(sum(H.^2,2)),1,classnum); 
   H = orth(H); 
end

% initialize alpha
alpha = ones(viewnum,1)*viewnum;

% initialize beta
beta = ones(viewnum,1)*(1/sqrt(viewnum));
    
% initialize W
B = cell(viewnum,1);
HX = cell(viewnum,1);
W = cell(viewnum,1);
for v = 1:viewnum 
    B{v} = X{v}'*X{v} + 10^-1*eye(d(v));
    HX{v} = H'*X{v};
    M = B{v}\(HX{v}'*HX{v});  
    W{v} = eig1(M,classnum,1,0);
    W{v} = W{v}*diag(1./sqrt(diag(W{v}'*B{v}*W{v})));
end    

%% =====================  Outer Updating ==========================
obj_o = [];
for iter_o = 1:NITER

% calculate lambda
if iter_o == 1
   K = 0; Z = 0;
   for v = 1:viewnum
       HU = H'*U{v};
       K = K + alpha(v)*(classnum-trace(HU*HU'));
       HXW = HX{v}*W{v};
       Z = Z + beta(v)*trace(HXW*HXW');
   end
   lambda = K/Z;  
else
   lambda = P/R;
end 

% calculate object value
obj_o = [obj_o; lambda];
if iter_o>=2 && obj_o(iter_o-1)-obj_o(iter_o)<10^-4
   break;
end

%% =====================  Inner Updating ==========================
obj_i = [];
for iter_i = 1:NITER  
    
% calculate object value
if iter_i == 1
   obj_i = [obj_i; 0];
else
   P = sum(alpha .* (G.^2));
   R = sum(beta .* Q);
   obj_i = [obj_i; P-lambda*R];
   if obj_i(iter_i-1)-obj_i(iter_i)<10^-4
      break;
   end
end

% update H
E = 0; F = 0;
for v = 1:viewnum 
    E = E + alpha(v)*(U{v}*U{v}');
    xw = X{v}*W{v};
    F = F + beta(v)*(xw*xw');
end
D = E + lambda*F;
H = eig1(D,classnum);

% update W
HX = cell(viewnum,1);
W = cell(viewnum,1);
for v = 1:viewnum 
    HX{v} = H'*X{v};
    M = B{v}\(HX{v}'*HX{v});  
    W{v} = eig1(M,classnum,1,0);
    W{v} = W{v}*diag(1./sqrt(diag(W{v}'*B{v}*W{v})));
end   

% update alpha
G = zeros(viewnum,1);
for v = 1:viewnum
    HU = H'*U{v};
    G(v) = sqrt(classnum-trace(HU*HU'));
end
alpha = zeros(viewnum,1);
for v = 1:viewnum
    alpha(v) = sum(G)/G(v);
end

% update beta
Q = zeros(viewnum,1);
for v = 1:viewnum
    HXW = HX{v}*W{v};
    Q(v) = trace(HXW*HXW');
end
beta = zeros(viewnum,1);
for v = 1:viewnum 
    beta(v) = Q(v)/(sqrt(sum(Q.^2)));
end

end

end

H = NormalizeFea(H,0);
result = kmeans(H, classnum, 'emptyaction', 'singleton', 'replicates', 100, 'display', 'off');

plot(obj_o);

end