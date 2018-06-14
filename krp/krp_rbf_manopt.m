% Implementation of BKRP without Slacks 
% Input: 
% X (required): an nxd matrix of features, n is the number of frames, d is the dimension of each feature.
% thresh (default 1/n): is the ranking threshold (eta in the paper).
% lambda (default 10n): : is the penalty constant on the order constraints.
% num_iter (default 100): is the number of iterations for the Riemannian gradient.
% sigma (default will be computed from data): bandwidth of the kernel matrix to be constructed.
% 
% implemented by Anoop Cherian, anoop.cherian@gmail.com
function [preimages, violate_ratio] = krp_rbf_manopt(X, thresh, lambda, num_iter, sigma)   
    [n,d] = size(X); 
    verbose = 0;

    z = initialize_z();
    if isempty(sigma)
        sigma = sqrt(mean(mypdist2(X, z, 'sqeuclidean')));
    end
    kernel = @(a,b) exp(-mypdist2(a,b,'sqeuclidean')/2/sigma^2);    
    
    manifold = euclideanfactory(1,d);          
    problem.M = manifold;
    problem.cost = @objective;
    problem.egrad = @grad;

    opts.maxiter = num_iter;
    opts.verbosity = 0; 
    kxz = kernel(X, z);
    vi = get_violations(kxz);
    violate_init = length(vi);
    if verbose
        figure(1);clf(1); plot(kxz);
        fprintf('before: norm(z)=%f num_violations=%d\n', norm(z), length(get_violations(kxz)));
    end
    z = conjugategradient(problem, z, opts);
                   
    kxz = kernel(X, z);
    vi = get_violations(kxz);
    violate_final = length(vi);
    if verbose
        figure(2);clf(2); plot(kxz);
        fprintf('final: norm(z)=%f num_violations=%d\n', norm(z), length(get_violations(kxz)));
    end
    violate_ratio = violate_final/violate_init;
    preimages = z;
       
    function obj = objective(zz)            
        kk = kernel(X, zz);
        vv = kk*ones(1,n) - ones(n,1)*kk' + thresh;
        vv = vv-diag(diag(vv));
        vv = triu(vv);
        vv(vv<=0)=0;
        obj = 0.5*(zz*zz') + lambda*sum(vv(:));    
    end

    function gZ = grad(zz)        
        kxz = kernel(X, zz);        
        [vi, vj] = get_violations(kxz);
        gZ = zz;
        if ~isempty(vi)    
            %numer =  sum(kxz(vi)'*X(vi,:) - kxz(vj)'*X(vj,:),1);
            uvi = unique(vi); uvj=unique(vj); 
            hvi = arrayfun(@(xx) nnz(vi==xx), uvi);
            hvj = arrayfun(@(xx) nnz(vj==xx), uvj);
            numer = sum((kxz(uvi).*hvi)'*X(uvi,:),1) - sum((kxz(uvj).*hvj)'*X(uvj,:),1);
            
            %C = sum(kxz(vi)-kxz(vj));  
            C = sum(kxz(uvi).*hvi) - sum(kxz(uvj).*hvj);
            gZ = gZ + lambda*(numer - zz*C)/sigma^2;        
        end
    end

    function [vi, vj] = get_violations(kxz) 
       vv = kxz*ones(1,n) - ones(n,1)*kxz' + thresh; %xi-xj+1 i<j
       vv = vv - diag(diag(vv));
       vv = triu(vv);
       vv(vv<=0)=0;
       ii = find(vv > 0);
       if any(ii)
           [vi,vj] = ind2sub(size(vv), ii);
       else
           [vi,vj] = deal([]);
       end
    end

   function xx = initialize_z()
        xx = mean(X,1);        
        %xx = krp_linear_manopt(X, thresh, 100, 100, sigma);          
        %[u,s,v]=svd(cov(X));
        %xx=u(:,1)';
    end
end