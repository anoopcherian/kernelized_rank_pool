% takes input nxd matrix. uses the multiplicative constraint than the
% additive one. xi <= beta/mu^2 xj. where xi are slacks.
function [preimages, violate_ratio] = krp_rbf_mult_withslack_manopt(X, thresh, lambda, num_iter, C, sigma)   
    [n,d] = size(X); 
    verbose = 0;

    z = initialize_z();
    if isempty(sigma)
        sigma = sqrt(mean(mypdist2(X, z, 'sqeuclidean')));
    end
    kernel = @(a,b) exp(-mypdist2(a,b,'sqeuclidean')/2/sigma^2);    
    
    % cache valid indices in the violation matrix
    idx = triu(ones(n,n)) - diag(ones(n,1));
    idx = find(idx);
    
    xi = (ones(1,n*(n-1)/2))*(thresh+thresh/10);
    
    manifold.Z = euclideanfactory(1,d);
    manifold.S = euclideanfactory(1,n*(n-1)/2);          
    problem.M = productmanifold(manifold);
    problem.cost = @objective;
    problem.egrad = @grad;

%     kxz = kernel(X, z);
%     [vi, vj] = get_violations(kxz,xi); %violate_init = length(vi);   
%     figure(1);clf(1); plot(kxz); figure(3); plot(xi); title('xi');
%     fprintf('before: norm(z)=%f num_violations=%d\n', norm(z), length(get_violations(kxz, xi)));        
%     
    opts.maxiter = num_iter;
    opts.verbosity = 0;     
    
    init.Z = z; init.S = xi;
    out = conjugategradient(problem, init, opts);
    z = out.Z; 
    xi = out.S;
    
%     kxz = kernel(X, z);
%     [vi, vj] = get_violations(kxz, xi); %violate_final = length(vi);   
%     figure(2);clf(2); plot(kxz); figure(4); plot(xi); title('xi');
%     fprintf('final: norm(z)=%f num_violations=%d\n', norm(z), length(get_violations(kxz,xi)));    

    violate_ratio = 0; %violate_final/violate_init;
    preimages = z;
       
 
    function obj = objective(M)  
        zz = M.Z; xi = M.S;
        kk = kernel(X, zz);       
        slack = ones(n,n); slack(idx) = xi.^2;
        vv = (thresh*kk*ones(1,n))./slack - ones(n,1)*kk';% - slack.^2;
        vv = vv(idx);        
        obj = 0.5*(zz*zz') + C*sum((1-xi.^2).^2) + lambda*sum(max(0,vv));        
    end

    function gM = grad(M)  
        zz = M.Z; xi = M.S;
        kxz = kernel(X, zz);        
        [vi, vj] = get_violations(kxz, xi);        
        gZ = zz;
        gXi = -4*C*xi.*(1-xi.^2);
        
        if ~isempty(vi)    
            slack = ones(n,n); slack(idx) = xi; 
            numer = 0;  ss = zeros(size(xi)); CC = 0; 
            for ii=1:length(vi)
                numer = numer + thresh*kxz(vi(ii))*(zz-X(vi(ii),:))/(slack(vi(ii),vj(ii))^2) - kxz(vj(ii))*(zz-X(vj(ii),:));                
                ss(vi(ii)) = kxz(vi(ii))/slack(vi(ii), vj(ii)).^3;
            end
            gZ = gZ + numer*lambda/sigma^2;
            gXi = gXi -2*lambda*thresh*ss;            
        end
        gM.Z = gZ; gM.S = gXi;
    end

    function [vi, vj] = get_violations(kxz, xi)
       slack = ones(n,n); slack(idx) = xi.^2;
       vv = thresh*kxz*ones(1,n)./slack - ones(n,1)*kxz'; % + thresh; %xi-xj+1 i<j
       %vv = vv(idx)-xi'.^2;
       vv=vv(idx);
       ii = idx(vv>0);
       if any(ii)
           [vi,vj] = ind2sub([n,n], ii);
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