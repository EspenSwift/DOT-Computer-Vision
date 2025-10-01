%% RelativePose_FromRED.m
% Compute relative pose data between RED and BLACK using only RED dataset
% Writes results to CSV

% --------- File save location ---------

%CHANGE THIS PARAMETER TO CHANGE WHERE THE CSV WRITES
outDir = '/Users/jamesmakhlouf/Desktop/UNIVERSITY/YEAR 4/Fall 2025/MAAE 4907/MAAE 4907 Q/Test Datasets/Soroush Test (LAR) Sept 29/';

outFile = 'TESTrelative_pose_data.csv'; % Change output file name
outputFile = fullfile(outDir, outFile);

RED = dataClass_rt; % dataClass_rt is the .mat file that contains all of the information
%% Relative Position Offsets:
%Update these with measured values
r_cam_x = 0; %m, offset in x from origin of Black, right camera
r_cam_y = 0; %m, offset in y from origin of Black, right camera

l_cam_x = 0; %m, offset in x from origin of Black, left camera
l_cam_y = 0.1; %m, offset in y from origin og Black, left camera

LAR_x = .30; %m, LAR offset in x from origin of Red
LAR_y =0; %m, LAR offset in y from origin of Red

%% DATA PROCESSING
% --------- Extract times ---------
tRed = RED.Time_s.Time; % Imported dataset as RED for common variable name

% Start at Red’s first valid time
t0 = tRed(1);
tEnd = tRed(end);
t = (t0 : mean(diff(tRed)) : tEnd)';

% --------- Resample all signals (all from RED) ---------
RED_Px   = resample(RED.RED_Px_m, t);
RED_Py   = resample(RED.RED_Py_m, t);
RED_Rz   = resample(RED.RED_Rz_rad, t);

BLACK_Px = resample(RED.BLACK_Px_m, t);
BLACK_Py = resample(RED.BLACK_Py_m, t);
BLACK_Rz = resample(RED.BLACK_Rz_rad, t);

N = numel(t);
LAR_Px = zeros(N,1);
LAR_Py = zeros(N,1);
CAM_Px = zeros(N,1);
CAM_Py = zeros(N,1);



% Compute rotated LAR and camera coordinates
% --------- Compute rotated LAR and camera coordinates ---------
for k = 1:N
    % Rotation matrices
    R_red = [cos(RED_Rz.Data(k))  -sin(RED_Rz.Data(k));
             sin(RED_Rz.Data(k))   cos(RED_Rz.Data(k))];
    R_black = [cos(BLACK_Rz.Data(k))  -sin(BLACK_Rz.Data(k));
               sin(BLACK_Rz.Data(k))   cos(BLACK_Rz.Data(k))];

    % LAR offset in Red frame → world
    offset_red = R_red * [LAR_x; LAR_y];
    LAR_Px(k) = RED_Px.Data(k) + offset_red(1);
    LAR_Py(k) = RED_Py.Data(k) + offset_red(2);

    % Camera offset in Black frame → world
    offset_black = R_black * [r_cam_x; r_cam_y]; % or l_cam_x/l_cam_y if left cam
    CAM_Px(k) = BLACK_Px.Data(k) + offset_black(1);
    CAM_Py(k) = BLACK_Py.Data(k) + offset_black(2);
end

% --------- Relative position: RED vs BLACK (already done) ---------
dx = RED_Px.Data - BLACK_Px.Data;
dy = RED_Py.Data - BLACK_Py.Data;
RelDist_center = sqrt(dx.^2 + dy.^2);

bearing_global_center = atan2(dy, dx);
Bearing_global_center_deg = rad2deg(bearing_global_center);
RelAngle_center_deg = rad2deg(bearing_global_center - BLACK_Rz.Data);

% --------- Relative position: LAR vs Camera ---------
dx_cam = LAR_Px - CAM_Px;
dy_cam = LAR_Py - CAM_Py;
RelDist_LARCam = sqrt(dx_cam.^2 + dy_cam.^2);

bearing_global_LARCam = atan2(dy_cam, dx_cam);
Bearing_global_LARCam_deg = rad2deg(bearing_global_LARCam);
RelAngle_LARCam_deg = rad2deg(bearing_global_LARCam - BLACK_Rz.Data);

% --------- Build table ---------
data = table(t, ...
             RelDist_center, Bearing_global_center_deg, RelAngle_center_deg, ...
             RelDist_LARCam, Bearing_global_LARCam_deg, RelAngle_LARCam_deg, ...
             'VariableNames', {'Time_s', ...
                               'RelDist_center_m','Bearing_center_deg','RelAngle_center_deg', ...
                               'RelDist_LARCam_m','Bearing_LARCam_deg','RelAngle_LARCam_deg'});

% --------- Write CSV ---------
writetable(data, outputFile);
fprintf('Relative pose data written to: %s\n', outputFile);
hold on
plot(t,RelDist_LARCam)
legend(num2str(LAR_y))