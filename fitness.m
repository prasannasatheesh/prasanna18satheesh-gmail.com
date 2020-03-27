function ft=fitness(dat,lab,sl)

ind=find(sl);
md=fitcensemble(dat(:,ind),lab);
LL=predict(md,dat(:,ind));
ft=1-(length(find(LL==lab))/length(LL));