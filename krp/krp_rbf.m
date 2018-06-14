% we assume feat = nxd matrix with n items each of dim d.
% thresh is the temporal ranking threshold.
% this variant uses single z, using objective:
% min_z 0.5*||z||^2 + max(0, k(x_i, z) +1 -k(x_j,z));
%
function preimages = krp_rbf(X, thresh, lambda, sigma)
    addpath(genpath('../tools/dists/'));
    n=size(X,1);
    kernel = @(a,b) exp(-mypdist2(a,b,'sqeuclidean')/2/sigma^2);    
    
    X = centralize_and_normalize_features(X);
   
    % the fixed point iterations!
    z = initialize_z();    
    sigma = sqrt(mean(mypdist2(X,z, 'sqeuclidean')));
    obj_old = 0; iter = 0; step_size = 0.1;
    
    kxz = kernel(X, z);
    obj = compute_obj(z);
    vi = get_violations(kxz);
    fprintf('iter = %d : obj = %f num_violations=%d norm_z=%f\n', iter, obj, length(vi), norm(z));
    while 1
        iter = iter + 1;
        kxz = kernel(X, z);
        plot(kxz); drawnow; pause(0.2);
        [vi, vj] = get_violations(kxz);
        
        if ~isempty(vi)
            numer =  sum(kxz(vi)'*X(vi,:) - kxz(vj)'*X(vj,:),1);
            C = sum(kxz(vi)-kxz(vj));
        else
            break;
        end
            
        if iter>20
            step_size = step_size/2;
        end
        grad_z = z + lambda*(numer - z*C)/sigma^2;            
        z_new = z - step_size*grad_z;         
                
        obj = compute_obj(z_new);
        fprintf('iter = %d : obj = %f norm(grad_z) = %f num_violations=%d norm_z=%f\n', iter, obj, norm(grad_z), length(vi), norm(z_new));
        if norm(grad_z)<1e-2 %abs(obj_old - obj)<1e-5
            break;
        end
        z = z_new;
        if iter> 1000
            break;
        end
        obj_old = obj;
    end
    
    kxz = kernel(X, z);
    plot(kxz); drawnow; pause(0.2);
    obj = compute_obj(z);
    vi = get_violations(kxz);
    fprintf('iter = %d : obj = %f num_violations=%d norm_z=%f\n', iter, obj, length(vi), norm(z_new));
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
        %xx = mean(X,1);
        xx = krp_linear(X, thresh, 100, sigma);       
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