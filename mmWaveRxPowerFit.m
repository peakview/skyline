X=-90:30:90;
Y=-90:20:90;
D=1:6;
[X2,Y2]=meshgrid(X,Y);
[X3,Y3,D3]=meshgrid(X,Y,D);

C=[-47,20,180,248];
RxPwr=C(1)-C(2)*log10(D3)+log10(cosd(X3/2).^C(3).*cosd(Y3/2).^C(4))+randn(size(D3))*0.1;
figure(1)
for ii=1:length(D)
subplot(length(D),1,ii)
contourf(X2,Y2,RxPwr(:,:,ii));
caxis([-120, -45]);
colorbar
end

[Pesti, fval, info, output] =  fsolve (@(P) (P(1)-P(2)*log10(D3)+log10(cosd(X3/2).^P(3).*cosd(Y3/2).^P(4)) - RxPwr), [0,0,0,0]);

%>> Pesti
%Pesti =
%
%   -47.008    20.014   179.883   247.983
