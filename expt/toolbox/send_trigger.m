function [delivery_receipt, trigger_log] = send_trigger(di,trigger,dt,latency)
    if(~exist('dt'))
        dt = 0.004; %4ms (2ms causes audio artifacts)
    end
    
     WaitSecs(latency);

    try
        DaqDOut(di,0,trigger);  %send trigger
        WaitSecs(dt);
        DaqDOut(di,0,0);  %clear trig
    catch ME
        disp(ME.message)
    end
    
    delivery_receipt = 'delivered';
    
end

