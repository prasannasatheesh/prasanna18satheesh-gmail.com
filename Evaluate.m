 function [EVAL cf] = Evaluate(ACTUAL,PREDICTED)
 cnames=['Classes'];
 un=unique(ACTUAL);
 for i=1:length(un)
idx = (ACTUAL()==i);

p = length(ACTUAL(idx));
n = length(ACTUAL(~idx));
N = p+n;

tp(i) = sum(ACTUAL(idx)==PREDICTED(idx));
tn(i) = sum(ACTUAL(~idx)==PREDICTED(~idx));
fp(i) = n-tn(i);
fn(i) = p-tp(i);

tp_rate = tp(i)/p;
tn_rate = tn(i)/n;

accuracy(i) = (tp(i)+tn(i))/N;
sensitivity(i) = tp_rate;
specificity(i) = tn_rate;
precision(i) = tp(i)/(tp(i)+fp(i));
recall(i) = sensitivity(i);
f_measure(i) = 2*((precision(i)*recall(i))/(precision(i) + recall(i)));
gmax(i) = sqrt(tp_rate*tn_rate);
sensitivity(isnan(sensitivity))=0;
precision(isnan(precision))=0;
recall(isnan(recall))=0;
f_measure(isnan(f_measure))=0;
gmax(isnan(gmax))=0;
 end
 EVAL = [max(accuracy) max(sensitivity) max(specificity) max(precision) max(recall) max(f_measure) max(gmax)];
 cf=[tp tn fp fn];