% The script presents the auditory MEG coref stimuli with comprehension
% questions.
%
% Input:
%   subject: 
%       The ID, typically with a session_ID in it: i.e FED_20XX_MEG
%   global_run: 
%       global_run number is mostly used to save the beh data to unique
%       csvs, so this is the accumulated run number across different tasks.
%       This will increment automatically after the first run is done.
%   debug (debug):
%       Run in debug mode
%
% Behavioral responses:
%   Response answers are collected.
%   Response time is collected.
%   Stim presentation is collected.
%   Stim condition is collected.
%   Fixation time is collected since that is variable.
%
% Trigger Design:
%   The triggers are defined in ./toolbox/load_triggerconstants.m.
%   Fixation, item start and end, and quiz start and end all get
%   unique triggers. Each unique item also gets a unique trigger
%   that is sent at the start of the item. Correct and incorrect
%   answers get distinct triggers that are sent
%   when users respond.

function MEG_coref(subject, global_run, debug)
    if ~(exist('debug') == 1)
        debug = 0;
    end
    PsychDebugWindowConfiguration();

    % Add helper functions
    addpath('./toolbox');

    Screen('Preference', 'SkipSyncTests', 1);

    % Prepare variables
    screens = Screen('Screens');
    screenNumber = max(screens);
    audio_latency = 0;
    screen_latency = 0.026;

    % Make screen smaller in debug mode
    [window, windowRect]=Screen('OpenWindow',screenNumber);

    % Connect to Daq if not specified
    try
        di = DaqDeviceIndex;
        DaqDConfigPort(di,0,0);
        DaqDOut(di,0,0);
    catch ME
        disp(ME.message)
    end

    % Psychtoolbox setup
    PsychDefaultSetup(2);
    screens = Screen('Screens');

    % Set up screen vars
    screenNumber = max(screens);
    black = BlackIndex(screenNumber);
    white = WhiteIndex(screenNumber);

    % Set the blend function for the screen
    Screen('BlendFunction', window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');

    % Set font vars
    Screen('TextSize', window, 40);
    Screen('TextFont', window, 'Courier');
    Screen('TextStyle', window, 0);
    
    % Triggers 
    load_trigger_constants;
    trigger_log = containers.Map;

    wiki_compQs = readtable('stim/wiki.csv');
    fourSent_compQs = readtable('stim/4sent.csv');

    % Show instructions
    DrawFormattedText(window, 'Press any button to continue', 'center', 'center', black);
    Screen(window, 'Flip');

    % Num repetitions per item (1)
    repetitions = 1;

    % Show fixation cross for 1s after subject presses to start item
    fix_delay = 1;

    wiki_items = randperm(15);
    fourSent_items = randperm(150);

    % Wait for user to proceed
    FlushEvents();
    get_user_response();

    % Get experiment start time
    t0 = GetSecs;

    n_runs = 11;

    for run = 1:n_runs
        if run == 1
            prefix = 'wiki_';
            name_map = wikiID2Name;
            items = wiki_items;
            compQs = wiki_compQs;
            item_ix = 1;
        else
            prefix = '4sent_';
            name_map = fourSentID2Name;
            items = fourSent_items;
            compQs = fourSent_compQs;
            if run == 2
                item_ix = 1;
            end
        end
        for run_pos=1:15 % 15 items per run
            item_id = items(item_ix);
            item_name = name_map(item_id);
            item_path = ['resources/' prefix num2str(item_id) '.wav'];

            % Make beh cells
            beh{1,1} = 'item';
            beh{1,2} = 'run';
            beh{1,3} = 'runpos';
            beh{1,4} = 'item_start_time';
            beh{1,5} = 'item_end_time';
            beh{1,6} = 'compQcorrect';
            beh{1,7} = 'RT';

            wavfilename = item_path;

            % Read WAV file from filesystem:
            [y, freq] = psychwavread(wavfilename);
            wavedata = y';
            nrchannels = size(wavedata,1); % Number of rows == number of channels.

            % Make sure we have always 2 channels stereo output.
            % Why? Because some low-end and embedded soundcards
            % only support 2 channels, not 1 channel, and we want
            % to be robust in our demos.
            if nrchannels < 2
                wavedata = [wavedata ; wavedata];
                nrchannels = 2;
            end

            device = [];

            % Perform basic initialization of the sound driver:
            InitializePsychSound;

            % Open the  audio device, with default mode [] (==Only playback),
            % and a required latencyclass of zero 0 == no low-latency mode, as well as
            % a frequency of freq and nrchannels sound channels.
            % This returns a handle to the audio device:
            try
                % Try with the 'freq'uency we wanted:
                pahandle = PsychPortAudio('Open', device, [], 0, freq, nrchannels);
            catch
                % Failed. Retry with default frequency as suggested by device:
                fprintf('\nCould not open device at wanted playback frequency of %i Hz. Will retry with device default frequency.\n', freq);
                fprintf('Sound may sound a bit out of tune, ...\n\n');

                psychlasterror('reset');
                pahandle = PsychPortAudio('Open', device, [], 0, [], nrchannels);
            end

            % Fill the audio playback buffer with the audio data 'wavedata':
            PsychPortAudio('FillBuffer', pahandle, wavedata);

            % Make beh cells
            beh{run_pos + 1,1} = item_name;
            beh{run_pos + 1,2} = run;
            beh{run_pos + 1,3} = run_pos;

            run_start = GetSecs;

            % Process triggers
            % Show fixation at beginning of run
            DrawFormattedText(window, '+', 'center', 'center', black);
            Screen('Flip', window);

            trigger = triggerName2ID('fixation');
            send_trigger(di, trigger, 0.004, screen_latency);
            trigger_log = update_trigger_log(trigger_log, trigger, GetSecs);

            while GetSecs <= run_start + fix_delay
                % While loop so fixation remains on screen for 4
                % seconds
            end

            % Start audio playback for 'repetitions' repetitions of the sound data,
            % start it immediately (0) and wait for the playback to start, return onset
            % timestamp.
            PsychPortAudio('Start', pahandle, repetitions, 0, 1);
            item_start = GetSecs - t0;

            % Process triggers
            % Item ID trigger
            trigger = triggerName2ID('item_start');
            send_trigger(di, trigger, 0.004, audio_latency);
            trigger_log = update_trigger_log(trigger_log, trigger, GetSecs);
            % Item start trigger
            trigger = triggerName2ID(item_name);
            send_trigger(di, trigger, 0.004, audio_latency);
            trigger_log = update_trigger_log(trigger_log, trigger, GetSecs);

            % Save start time
            beh{run_pos + 1,4} = item_start;

            % Stop playback:
            if debug == 1
                % Only play 1s
                WaitSecs(1);
                PsychPortAudio('Stop', pahandle);
            else
                % Play whole file
                PsychPortAudio('Stop', pahandle, 3);
            end

            % Save end time
            item_end = GetSecs - t0;
            beh{run_pos + 1,5} = item_end;

            % Close the audio device:
            PsychPortAudio('Close', pahandle);

            % Process triggers
            trigger = triggerName2ID('item_end');
            send_trigger(di, trigger, 0.004, audio_latency);
            trigger_log = update_trigger_log(trigger_log, trigger, GetSecs);

            if rand < 0.5 % ~1/2 get comp Qs
                % Select comprehension Qs for this item
                T = compQs(compQs.ItemID == item_id,:);

                % Process triggers
                trigger = triggerName2ID('quiz_start');
                send_trigger(di, trigger, 0.004, screen_latency);
                trigger_log = update_trigger_log(trigger_log, trigger, GetSecs);

                % Present each comp Q and record response
                if height(T) > 0
                    q = 1;

                    % Display question and answers
                    correct_answer = T.CompQOptCorrect{q};
                    incorrect_answer = T.CompQOptIncorrect{q};
                    if rand < 0.5
                        answer_a = correct_answer;
                        answer_b = incorrect_answer;
                        answer = 'A';
                    else
                        answer_a = incorrect_answer;
                        answer_b = correct_answer;
                        answer = 'B';
                    end

                    q_text = [T.CompQ{q} '\n\nA.    ' answer_a '\n\nB.    ' answer_b];
                    DrawFormattedText(window, q_text, 200, 'center', black, 50);
                    Screen('Flip', window);

                    % Capture user response
                    FlushEvents();
                    [char, responseTime] = get_user_response();

                    A_chars = [11, 16]; % Left side, right side button 1 (blue)
                    B_chars = [12, 17]; % Left side, right side button 2 (yellow)

                    if ((A_chars(1) == char || A_chars(2) == char) && strcmp(answer, 'A')) || ((B_chars(1) == char || B_chars(2) == char) && strcmp(answer, 'B'))
                        correct = 1;
                    else
                        correct = 0;
                    end

                    % Process triggers
                    if correct == 1
                        trigger = triggerName2ID('correct');
                    else
                        trigger = triggerName2ID('incorrect');
                    end
                    send_trigger(di, trigger, 0.004, audio_latency);
                    trigger_log = update_trigger_log(trigger_log, trigger, GetSecs);

                    % Save question score
                    beh{run_pos + 1,6} = correct;
                    beh{run_pos + 1,7} = responseTime;
                end

                % Process triggers
                trigger = triggerName2ID('quiz_end');
                send_trigger(di, trigger, 0.004, audio_latency);
                trigger_log = update_trigger_log(trigger_log, trigger, GetSecs);

            end

            % Increment item index
            item_ix = item_ix + 1;

        end

        % Save data after every run
        sdir = fullfile('./Output');
        cell2csv(beh, fullfile(sdir,[subject '_run' num2str(global_run) '.csv']));
        save(fullfile(sdir,['triggers_' subject '_run' num2str(global_run) '.mat']),'trigger_log');

        if run < n_runs
            % Wait for user to continue
            DrawFormattedText(window, 'You may take a short rest now.', 'center', 'center', black);
            Screen('Flip', window);
            WaitSecs(5);
            DrawFormattedText(window, 'Press any button to continue.', 'center', 'center', black);
            Screen('Flip', window);
            FlushEvents();
            get_user_response();

            % Increment global_run
            global_run = global_run + 1;
        end

    end

    DrawFormattedText(window, 'Thank you!', 'center', 'center', black);
    Screen('Flip', window);
    FlushEvents();
    get_user_response();

    sca;
end

