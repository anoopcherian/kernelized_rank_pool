% takes input nxd matrix.
% solves min_z 0.5\sum_i ||x_i - z||^2 + \lambda \sum_{i<j} max(0, k(x_i, z) + 1 - k(x_j, z)).
% Implementation of IBKRP with Slack variables. 
% Input: 
% X (required): an nxd matrix of features, n is the number of frames, d is the dimension of each feature.
% thresh (default 1/n): is the ranking threshold (eta in the paper).
% lambda (default 10n): : is the penalty constant on the order constraints.
% num_iter (default 100): is the number of iterations for the Riemannian gradient.
% sigma (default will be computed from data): bandwidth of the kernel matrix to be constructed.
% C is the penalty on the slacks.
% 
% outputs dx1 vector which corresponds to the pooled pre-image.
%
% implemented by Anoop Cherian, anoop.cherian@gmail.com
function [preimages, violate_ratio] = krp_rbf_Xreconstruct_withslack_manopt(X, thresh, lambda, C, num_iter, sigma)   
    [n,d] = size(X); 

    preimages = 0;
    sqX = sum(sum(X.^2));
    z = initialize_z();
    if isempty(sigma)
        sigma = sqrt(mean(mypdist2(X, z, 'sqeuclidean')));
    end
    if isempty(thresh)
        thresh = 1/(1*n);
    end
    if isempty(lambda)
        lambda = 1*n;
    end
                
    % cache valid indices in the violation matrix
    idx = triu(ones(n,n)) - diag(ones(n,1));
    idx = find(idx);
    
    xi = 0.5*ones(1,n*(n-1)/2);
    
    kernel = @(XX,zz) exp(-sum(bsxfun(@minus, XX, zz).^2,2)/2/sigma^2);    
    
    manifold.Z = euclideanfactory(1,d);
    manifold.S = euclideanfactory(1,n*(n-1)/2);
    problem.M = productmanifold(manifold);
    problem.cost = @objective;
    problem.egrad = @grad;
        
    %checkgradient(problem); return;

    opts.maxiter = num_iter;
    opts.verbosity = 0; 
%     kxz = kernel(X, z);
%     [vi, vj] = get_violations(kxz,xi); violate_init = length(vi);   
%     figure(1);clf(1); plot(kxz); figure(3); plot(xi); title('xi');
%     fprintf('before: norm(z)=%f num_violations=%d\n', norm(z), length(get_violations(kxz, xi)));
    init.Z = z; init.S = xi;
    out = conjugategradient(problem, init, opts);
    z = out.Z;
    xi = out.S;
                   
%     kxz = kernel(X, z);
%     [vi, vj] = get_violations(kxz, xi); violate_final = length(vi);   
%     figure(2);clf(2); plot(kxz); figure(4); plot(xi); title('xi');
%     fprintf('final: norm(z)=%f num_violations=%d\n', norm(z), length(get_violations(kxz,xi)));
    violate_ratio = 0; %violate_final/violate_init;
    preimages = z;
       
    
    function obj = objective(M)
        zz = M.Z; xi = M.S;
        kk = kernel(X, zz);
        %slack = zeros(n,n); slack(idx) = xi;
        vv = kk*ones(1,n) - ones(n,1)*kk' + thresh;% - slack.^2;
        vv = vv(idx);
        obj = 0.5*(sqX + n*(zz*zz')-2*sum(X*zz')) + C*sum(xi.^2) + lambda*sum(max(0,vv-xi'.^2));     % ||X-z||^2 + lambda*violoations.        
        %obj = 0.5*sum(sum((X-ones(n,1)*zz).^2)) + lambda*sum(vv(:));
    end

    function [gM] = grad(M)        
        zz = M.Z; xi = M.S;
        kxz = kernel(X, zz);        
        [vi, vj, vxi] = get_violations(kxz, xi);
        gZ = -sum((X-ones(n,1)*zz),1);     
        gXi = 2*C*xi;
         
        if ~isempty(vi)  
            uvi = unique(vi); uvj=unique(vj);
            if length(uvi)>1, hvi = hist(vi, uvi)';  else hvi = length(vi); end
            if length(uvj)>1, hvj = hist(vj, uvj)';  else hvj = length(vj); end
 
            numer = sum((kxz(uvi).*hvi)'*X(uvi,:),1) - sum((kxz(uvj).*hvj)'*X(uvj,:),1);
            CC = sum(kxz(uvi).*hvi) - sum(kxz(uvj).*hvj);  
            gZ = gZ + lambda*(numer - zz*CC)/sigma^2;                    
            
            gXi = gXi - 2*lambda*vxi;
        end
        gM.Z = gZ; gM.S = gXi;
    end

    function [vi, vj, vxi] = get_violations(kxz, xi) 
       %slack = zeros(n,n); slack(idx) = xi;
       vv = kxz*ones(1,n) - ones(n,1)*kxz' + thresh; %xi-xj+1 i<j              
       vv = vv(idx)-xi'.^2;
       ii = idx(vv>0);
       if any(ii)
           [vi,vj] = ind2sub([n,n], ii);
       else
           [vi,vj] = deal([]);
       end
       vxi = xi; vxi(vv<0) = 0;
    end

   function xx = initialize_z()
        %xx = median(X,1);        
        xx = X(1,:);        
       
        %xx = krp_linear_manopt(X, thresh, 10, 100, sigma);          
        %[u,s,v]=svd(cov(X));
        %xx=u(:,1)';
    end
end