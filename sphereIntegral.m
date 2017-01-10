% integral over sphere surface in polar coordination

clear
% area_sum=0; % area_sum
% step=0.0003;
% aods=0:step:2*pi;
% zods=0:step:pi;
% for iiaod=1:length(aods)
%     aod=aods(iiaod);
%     for iizod=1:length(zods)
%         zod=zods(iizod);
%         area_sum=area_sum+sin(zod);
%     end
% end
% area_sum*step*step/4


area_sum=0; % area_sum
step=0.0003;
aods=0:step:2*pi;
zods=0:step:pi;
for iiaod=1:length(aods)
    aod=aods(iiaod);
    area_sum=area_sum+sum(sin(zods));
end
area_sum*step*step/4