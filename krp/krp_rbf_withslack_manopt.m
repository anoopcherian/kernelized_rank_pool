% Implementation of BKRP with Slack variables. 
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
%
function [preimages, violate_ratio] = krp_rbf_withslack_manopt(X, thresh, lambda, num_iter, C, sigma)   
    [n,d] = size(X); 
    verbose = 0;

    preimages = 0;
    z = initialize_z();
    if isempty(sigma)
        sigma = sqrt(mean(mypdist2(X, z, 'sqeuclidean')));
    end
    kernel = @(a,b) exp(-mypdist2(a,b,'sqeuclidean')/2/sigma^2);    
    
    % cache valid indices in the violation matrix
    idx = triu(ones(n,n)) - diag(ones(n,1));
    idx = find(idx);
    
    xi = mean(mean(X))*ones(1,n*(n-1)/2)/100;
    
    manifold.Z = euclideanfactory(1,d);
    manifold.S = euclideanfactory(1,n*(n-1)/2);          
    problem.M = productmanifold(manifold);
    problem.cost = @objective;
    problem.egrad = @grad;
        
    if verbose == 1
        kxz = kernel(X, z);
        [vi, vj] = get_violations(kxz,xi); violate_init = length(vi);   
        figure(1);clf(1); plot(kxz); figure(3); plot(xi); title('xi');
        fprintf('before: norm(z)=%f num_violations=%d\n', norm(z), length(get_violations(kxz, xi)));     
    end
    
    opts.maxiter = num_iter;
    opts.verbosity = 0;     
    
    init.Z = z; init.S = xi;
    out = conjugategradient(problem, init, opts);
    z = out.Z; 
    xi = out.S;
    
    if verbose == 1
        kxz = kernel(X, z);
        [vi, vj] = get_violations(kxz, xi); violate_final = length(vi);   
        figure(2);clf(2); plot(kxz); figure(4); plot(xi); title('xi');
        fprintf('final: norm(z)=%f num_violations=%d\n', norm(z), length(get_violations(kxz,xi))); 
    end

    violate_ratio = 0; %violate_final/violate_init;
    preimages = z;
        
    function obj = objective(M)  
        zz = M.Z; xi = M.S;
        kk = kernel(X, zz);        
        vv = kk*ones(1,n) - ones(n,1)*kk' + thresh;% - slack.^2;
        vv = vv(idx);        
        obj = 0.5*(zz*zz') + C*sum(xi.^2) + lambda*sum(max(0,vv-xi'.^2));    
    end

    function gM = grad(M)  
        zz = M.Z; xi = M.S;
        kxz = kernel(X, zz);        
        [vi, vj, vxi] = get_violations(kxz, xi);        
        gZ = zz;
        gXi = 2*C*xi;
        
        if ~isempty(vi)    
            %numer =  sum(kxz(vi)'*X(vi,:) - kxz(vj)'*X(vj,:),1);
            uvi = unique(vi); uvj=unique(vj); 
            if length(uvi)>1, hvi = hist(vi, uvi)';  else hvi = length(vi); end
            if length(uvj)>1, hvj = hist(vj, uvj)';  else hvj = length(vj); end
            numer = sum((kxz(uvi).*hvi)'*X(uvi,:),1) - sum((kxz(uvj).*hvj)'*X(uvj,:),1);
            
            %C = sum(kxz(vi)-kxz(vj));  
            CC = sum(kxz(uvi).*hvi) - sum(kxz(uvj).*hvj);
            gZ = gZ + lambda*(numer - zz*CC)/sigma^2;   
            gXi = gXi - 2*lambda*vxi;
        end
        gM.Z = gZ; gM.S = gXi;
    end

    function [vi, vj, vxi] = get_violations(kxz, xi) 
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
        xx = mean(X,1);        
        %xx = krp_linear_manopt(X, thresh, 100, 100, sigma);          
        %[u,s,v]=svd(cov(X));
        %xx=u(:,1)';
    end
end