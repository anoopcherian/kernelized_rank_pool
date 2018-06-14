% we assume feat = nxd matrix with n items each of dim d.
% thresh is the temporal ranking threshold.
% this variant uses single z, using objective:
% min_z ||z||^2 + max(0, k(x_i, z) +1 -k(x_j,z)), where k(a,b) = a'b.
% (linear case)
%
function preimages = krp_linear(X, thresh, lambda, sigma)
    addpath(genpath('../tools/dists/'));
    n=size(X,1);    
    kernel = @(a,b) a*b';
    
    X = centralize_and_normalize_features(X);
       
    z = initialize_z();
    obj_old = 0; iter = 0; step_size = 0.0001;
    
    kxz = kernel(X, z);
    obj = compute_obj(z);
    vi = get_violations(kxz);
    fprintf('iter = %d : obj = %f num_violations=%d norm_z=%f\n', iter, obj, length(vi), norm(z));
    while 1
        iter = iter + 1;
        kxz = kernel(X, z);
        %plot(kxz); drawnow; pause(0.2);
        [vi, vj] = get_violations(kxz);
        
        if ~isempty(vi)    
            numer = lambda*sum(X(vi,:) - X(vj,:),1);            
        else
            break;
        end
            
        z_new = z - step_size*numer;       
        
        obj = compute_obj(z_new);
        fprintf('linear: iter = %d : obj = %f num_violations=%d norm_z=%f\n', iter, obj, length(vi), norm(z_new));
        if abs(obj_old - obj)<1e-5
            break;
        end
        z = z_new;
        if iter> 1000
            break;
        end
        obj_old = obj;
    end
    
    kxz = kernel(X, z);
    obj = compute_obj(z);
    vi = get_violations(kxz);
    fprintf('linear: iter = %d : obj = %f num_violations=%d norm_z=%f\n', iter, obj, length(vi), norm(z));
    preimages = z;
    %plot(z);
    return;
    
    % kk is a vector.
    function [vi,vj] = get_violations(kxz)
        [vi,vj] = deal([]); cnt = 0;
        for i=1:n
            for j=i+1:n
                if kxz(i)+thresh>kxz(j)
                    cnt = cnt + 1;
                    vi(cnt) = i; vj(cnt) = j;
                end
            end
        end
    end
   
    function xx = initialize_z()
        xx = mean(X,1);
        %[u,s,v]=svd(cov(X));
        %xx=u(:,1)';
    end

    function obj = compute_obj(zz)
        kk = kernel(X,zz);
        vv = kk*ones(1,n) - ones(n,1)*kk' + thresh;
        vv = vv-diag(diag(vv));
        vv = triu(vv);
        vv(vv<=0)=0;
        obj = 0.5*(zz*zz') + lambda*sum(vv(:));
    end

    % assumes X are row vectors
    function X = centralize_and_normalize_features(X)
        %X = X - ones(n,1)*mean(X,1);
        %X = normr(X);
    end
end