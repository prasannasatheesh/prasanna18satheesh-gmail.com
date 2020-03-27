clc;clear all;close all;
[f p]=uigetfile('*');
X=importdata([p f]);
data=X.data;
data2=data(:,1:end-1);class=data(:,end);
data1=knnimpute(data2);

%%%%%%%%Feature selection 
FSL=0;FSU=1;
D=size(data1,2);
for i=1:10
FS(i,:)=FSL+randi([0 1],[1 D])*(FSU-FSL);
try
fit(i)=fitness(data1,class,FS(i,:));
catch
    fit(i)=1;
    continue;
end
end
ind=find(fit==min(fit));
FSnew=FS(ind,:);
pdp=0.1;
row=1.204;V=5.25;S=0.0154;cd=0.6;CL=0.7;hg=1;sf=18;
Gc=1.9;
D1=1/(2*row*V.^2*S*cd);L=1/(2*row*V.^2*S*CL);
tanpi=D1/L;dg=hg/(tanpi*sf);aa=randi([1 length(ind)]);
iter=1;maxiter=2;
while(iter<maxiter)
for i=1:10
if(rand>=pdp)
    FS(i,:)=round(FS(i,:)+(dg*Gc*abs(FSnew(1,:)-FS(i,:))));
else
   FS(i,:)=FSL+randi([0 1],[1 D])*(FSU-FSL);
 
end
Fh=FS;
fit1(i)=fitness(data1,class,FS(i,:));
ind1=find(fit1==min(fit1));
FSnew1=FS(ind1,:);
if(rand>pdp)
    FS(i,:)=round(FS(i,:)+(dg*Gc*abs(FSnew(aa,:)-FS(i,:))));
else
   FS(i,:)=FSL+randi([0 1],[1 D])*(FSU-FSL);
 
end
Fa=FS;
fit2(i)=fitness(data1,class,FS(i,:));
ind2=find(fit2==min(fit2));
FSnew2=FS(ind2,:);
end
Sc=sqrt(sum(abs(Fh-Fa)).^2);
Smin=(10*exp(-6))/(365).^(iter/(maxiter/2.5));
if(Sc<Smin)
    season=summer;
    for i=1:10
        FS(i,:)=FSL+levy(1,D,1.5)*(FSU-FSL);
    end
    
else
    season=winter;
    break;
end
%%%Searching method
fit3(i)=fitness(data1,class,FS(i,:));
ind3=find(fit3==min(fit3));
final=abs(round([Fh(ind1,:);Fa(ind2,:);FS(ind3,:)]));
for i=1:size(final,1)
    fitt(i)=fitness(data1,class,final(i,:));
end
best(iter)=min(fitt);
[ff inn]=min(fitt);
bestfeat(iter,:)=final(inn,:);pdp=best(iter);

iter=iter+1;
end
sel=find(bestfeat(end,:));
disp('Selected Features');disp(sel)
dataA =data2(:,sel);  % some test data
p = .7 ;     % proportion of rows to select for training
N = size(dataA,1);  % total number of rows 
tf = false(N,1);   % create logical index vector
tf(1:round(p*N)) = true;   
tf = tf(randperm(N));   % randomise order
dataTraining = dataA(tf,:);labeltraining=class(tf);
dataTesting = dataA(~tf,:);labeltesting=class(~tf);
disp('Training feature size');disp(length(dataTraining))
disp('Testing feature size');disp(length(dataTesting))


svt=svmtrain(dataTraining,labeltraining);
out1=svmclassify(svt,dataTesting);
mdl=fitcknn(dataTraining,labeltraining);
out2=predict(mdl,dataTesting);
%%%%%%%  NB %%%%%%%%
mdl=fitcensemble(dataTraining,labeltraining);
out3=predict(mdl,dataTesting);
tp=length(find(out3==labeltesting));
msgbox([{['Out of ',num2str(length(out3))]},{[num2str(tp),'are correctly classified']}])
delete(gcp('nocreate'))
disp('%%%%%%%%  KNN %%%%%%%%%%%%%%')
[EVAL CF] = Evaluate(out2,labeltesting);
disp('Accuracy (%)');disp(EVAL(1)*100);
disp('Precision (%)');disp(EVAL(4)*100);
disp('Recall (%)');disp(EVAL(5)*100);
disp('Fmeasure (%)');disp(EVAL(6)*100);

disp('True Positive');disp(CF(1))
disp('True Negative');disp(CF(2))
disp('False Positive');disp(CF(3))
disp('False Negative');disp(CF(4))
disp('%%%%%%%%  SVM %%%%%%%%%%%%%%')
[EVAL3 CF] = Evaluate(out1,labeltesting);
disp('Accuracy (%)');disp(EVAL3(1)*100);
disp('Precision (%)');disp(EVAL3(4)*100);
disp('Recall (%)');disp(EVAL3(5)*100);
disp('Fmeasure (%)');disp(EVAL3(6)*100);
 
disp('True Positive');disp(CF(1))
disp('True Negative');disp(CF(2))
disp('False Positive');disp(CF(3))
disp('False Negative');disp(CF(4))

disp('%%%%%%  NB %%%%%%%%%%%%%%')
[EVAL2 CF] = Evaluate(out3,labeltesting);
disp('Accuracy (%)');disp(EVAL2(1)*100);
disp('Precision (%)');disp(EVAL2(4)*100);
disp('Recall (%)');disp(EVAL2(5)*100);
disp('Fmeasure (%)');disp(EVAL2(6)*100);

disp('True Positive');disp(CF(1))
disp('True Negative');disp(CF(2))
disp('False Positive');disp(CF(3))
disp('False Negative');disp(CF(4))
