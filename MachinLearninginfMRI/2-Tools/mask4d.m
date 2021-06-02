function [Y,inds] = mask4d(X,mask)
inds= find(mask==1);
[Nx,Ny,Nz,Nt] = size(X);
for i=1:Nt
    tmp = X(:,:,:,i);
    Y(:,i) = tmp(inds);
end
end
