function y = getFFT(x,s,name)  %UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
fs =100;                % Sample frequency (Hz)
m = length(x);          % Window length
n = pow2(nextpow2(m));  % Transform length
y = fft(x,n);           % DFT
f = (0:n-1)*(fs/n);     % Frequency range
power = y.*conj(y)/n;   % Power of the DFT

subplot(3,1,s)
%semilogy(f,power);
[pks,loc]=findpeaks(power,f);
semilogy(loc,pks);
xlabel('Frequency (Hz)')
ylabel('Power')
legend(name)


end

