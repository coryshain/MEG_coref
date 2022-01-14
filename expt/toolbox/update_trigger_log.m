% This function takes in a collections.map object and updates the
% trigger_times array corresponding to the trigger_string
% Input:
%   -trigger_log is of type collections.map
%   -key is the trigger value
%   -timestamp is the timestamp of the respective key
% Output:
% An updated trigger_log

function trigger_log = update_trigger_log(trigger_log, key, timestamp)
    % key must be char
    key=num2str(key);
    if ~isKey(trigger_log,key)
        trigger_log(key) = [timestamp];
    else
        trigger_log(key) = [trigger_log(key) timestamp]; 
    end
end