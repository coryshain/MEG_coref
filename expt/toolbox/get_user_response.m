function [keyCode, responseTime] = get_user_response()
    %disable output to the command window
    ListenChar(2);

    %read the current time on the clock
    startTime = GetSecs;
    %read the keyboard
    [ endTime, keyCode, ~ ] = KbWait([], 2); % forWhat=3 waits for key release
    responseTime = endTime - startTime;
    %enable output to the command window
    ListenChar(0);
    
    keyCode = find(keyCode);
    
end