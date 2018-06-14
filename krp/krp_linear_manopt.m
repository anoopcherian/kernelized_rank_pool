% takes input nxd matrix.
% min_z 0.5*||z||^2 + lambda \sum_{i<j} max(0, z'x_i + 1 - z'x_j).
function [preimages, violate_ratio] = krp_linear_manopt(X, thresh, lambda, num_iter, sigma, show_stuff)   
    [n,d] = size(X); 

    if nargin == 5
        show_stuff = 0;
    end
    kernel = @(a,b) a*b';
    
    manifold = spherefactory(1,d);          
    problem.M = manifold;
    problem.cost = @objective;
    problem.egrad = @grad;
      
    % cache valid indices in the violation matrix
    idx = triu(ones(n,n)) - diag(ones(n,1));
    idx = find(idx);
    
    %checkgradient(problem); Z = rand; return;
    z = initialize_z();
    
    opts.maxiter = num_iter;
    opts.verbosity = 0; 
    kxz = kernel(X, z);
    vi = get_violations(kxz);
    violate_init = length(vi);
    if show_stuff == 1        
        figure(1);clf(1); plot(kxz);
        fprintf('before: norm(z)=%f num_violations=%d\n', norm(z), length(get_violations(kxz)));
    end
    z = conjugategradient(problem, z, opts);
    
    kxz = kernel(X, z);
    vi = get_violations(kxz);
    violate_final = length(vi);
    if show_stuff == 1                       
        figure(2);clf(2); plot(kxz);
        fprintf('final: norm(z)=%f num_violations=%d\n', norm(z), length(get_violations(kxz)));
    end
    violate_ratio = violate_final/violate_init;
    preimages = z;
       
    function obj = objective(zz)            
        kk = kernel(X, zz);
        vv = kk*ones(1,n) - ones(n,1)*kk' + thresh;
        vv = vv(idx);
        obj = 0.5*(zz*zz') + lambda*sum(sum(vv));    
    end

    function gZ = grad(zz)        
        kxz = kernel(X, zz);        
        [vi, vj] = get_violations(kxz);
        
        gZ = zz;
        if ~isempty(vi)    
            uvi = unique(vi); uvj=unique(vj); 
            if length(uvi)>1, hvi = hist(vi, uvi)';  else hvi = length(vi); end
            if length(uvj)>1, hvj = hist(vj, uvj)';  else hvj = length(vj); end
            gZ = gZ + lambda*(sum(hvi'*X(uvi,:),1) - sum(hvj'*X(uvj,:),1));                 
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
        xx = mean(X,1);
        %[u,s,v]=svd(cov(X));
        %xx=u(:,1)';
    end
end