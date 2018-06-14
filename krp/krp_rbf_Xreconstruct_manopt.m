% takes input nxd matrix.
% solves min_z 0.5\sum_i ||x_i - z||^2 + \lambda \sum_{i<j} max(0, k(x_i, z) + 1 - k(x_j, z)).
function [preimages, violate_ratio] = krp_rbf_Xreconstruct_manopt(X, thresh, lambda, num_iter, sigma)   
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
    
    kernel = @(XX,zz) exp(-sum(bsxfun(@minus, XX, zz).^2,2)/2/sigma^2);    
    
    manifold = euclideanfactory(1,d);          
    problem.M = manifold;
    problem.cost = @objective;
    problem.egrad = @grad;
        
    %checkgradient(problem); return;

    opts.maxiter = num_iter;
    opts.verbosity = 0; 
%     kxz = kernel(X, z);
%     [vi, vj] = get_violations(kxz); violate_init = length(vi);   
%     figure(1);clf(1); plot(kxz);
%     fprintf('before: norm(z)=%f num_violations=%d\n', norm(z), length(get_violations(kxz)));
    z = conjugategradient(problem, z, opts);
                   
%     kxz = kernel(X, z);
%     [vi, vj] = get_violations(kxz); violate_final = length(vi);   
%     figure(2);clf(2); plot(kxz);
%     fprintf('final: norm(z)=%f num_violations=%d\n', norm(z), length(get_violations(kxz)));
    violate_ratio = 0; %violate_final/violate_init;
    preimages = z;
       
    
    function obj = objective(zz)        
        kk = kernel(X, zz);
        vv = kk*ones(1,n) - ones(n,1)*kk' + thresh;
        vv = vv(idx);
        obj = 0.5*(sqX + n*(zz*zz')-2*sum(X*zz')) + lambda*sum(max(0,vv));     % ||X-z||^2 + lambda*violoations.        
        %obj = 0.5*sum(sum((X-ones(n,1)*zz).^2)) + lambda*sum(vv(:));
    end

    function gZ = grad(zz)        
        kxz = kernel(X, zz);        
        [vi, vj] = get_violations(kxz);
        gZ = -sum((X-ones(n,1)*zz),1);        
         
        if ~isempty(vi)  
            uvi = unique(vi); uvj=unique(vj);
            if length(uvi)>1, hvi = hist(vi, uvi)';  else hvi = length(vi); end
            if length(uvj)>1, hvj = hist(vj, uvj)';  else hvj = length(vj); end
 
            numer = sum((kxz(uvi).*hvi)'*X(uvi,:),1) - sum((kxz(uvj).*hvj)'*X(uvj,:),1);
            C = sum(kxz(uvi).*hvi) - sum(kxz(uvj).*hvj);  
            gZ = gZ + lambda*(numer - zz*C)/sigma^2;        
        end
    end

    function [vi, vj] = get_violations(kxz) 
       vv = kxz*ones(1,n) - ones(n,1)*kxz' + thresh; %xi-xj+1 i<j
       vv = vv(idx);
       ii = idx(vv>0);
       if any(ii)
           [vi,vj] = ind2sub([n,n], ii);
       else
           [vi,vj] = deal([]);
       end
    end

   function xx = initialize_z()
        %xx = median(X,1);        
        xx = X(1,:);        
       
        %xx = krp_linear_manopt(X, thresh, 10, 100, sigma);          
        %[u,s,v]=svd(cov(X));
        %xx=u(:,1)';
    end
end