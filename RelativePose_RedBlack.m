%% RelativePose_LAR_CAM.m
% Compute relative pose data between RED and BLACK using only RED dataset
% Writes results to CSV
% Note: Chaser (vision) is red, LAR (target) is black
% --------- File save location ---------

%CHANGE THIS PARAMETER TO CHANGE WHERE THE CSV WRITES
outDir = '/outDir/'; %copy as path
outFile = 'LAR-CAM Relative Pose Data.xlsx'; % Change output file name
outputFile = fullfile(outDir, outFile);

RED = dataClass_rt; % dataClass_rt is the .mat file that contains all of the information
%% Relative Position Offsets:
%These are rough values that were measured using a tape measurer -
%estimates are from geometrical centre of chaser and target.

% NOTE:

% cam_x numbers depend on where the sensor of the camera actually as - the
% best estimate from the lens is anywhere within the camera unit itself
% which is about 1 inch wide. Using these numbers, the values for cam_x
% could range from 0.127 to 0.138 m (0.125- 0.140 m)

% On the chaser, the positive x axis passes through the camera
% The positive y axis passes through the air pressure gauge face


r_cam_x = 0.127; %m, offset in x from origin of Red, right camera
r_cam_y = -0.1; %m, offset in y from origin of Red, right camera

l_cam_x = 0.127; %m, offset in x from origin of Red, left camera (This value 
l_cam_y = 0.02875; %m, offset in y from origin og Red, left camera

LAR_x = 0.2286; %m, LAR offset in x from origin of Black
LAR_y =0; %m, LAR offset in y from origin of Black

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
r_CAM_Px = zeros(N,1);
r_CAM_Py = zeros(N,1);
l_CAM_Px = zeros(N,1);
l_CAM_Py = zeros(N,1);


% Compute rotated LAR and camera coordinates
% --------- Compute rotated LAR and camera coordinates ---------
for k = 1:N
    % Rotation matrices
    R_red = [cos(RED_Rz.Data(k))  -sin(RED_Rz.Data(k));
             sin(RED_Rz.Data(k))   cos(RED_Rz.Data(k))];
    R_black = [cos(BLACK_Rz.Data(k))  -sin(BLACK_Rz.Data(k));
               sin(BLACK_Rz.Data(k))   cos(BLACK_Rz.Data(k))];

    % LAR offset in Black frame → world
    offset_black = R_black * [LAR_x; LAR_y];
    LAR_Px(k) = BLACK_Px.Data(k) + offset_black(1);
    LAR_Py(k) = BLACK_Py.Data(k) + offset_black(2);

    % Camera offset in Red frame → world
    r_offset_red = R_red * [r_cam_x; r_cam_y]; % right cam
    r_CAM_Px(k) = RED_Px.Data(k) + r_offset_red(1);
    r_CAM_Py(k) = RED_Py.Data(k) + r_offset_red(2);

    l_offset_red = R_red * [l_cam_x; l_cam_y]; % left cam
    l_CAM_Px(k) = RED_Px.Data(k) + l_offset_red(1);
    l_CAM_Py(k) = RED_Py.Data(k) + l_offset_red(2);
end

% --------- Relative position: RED vs BLACK  ---------
dx = BLACK_Px.Data - RED_Px.Data;
dy = BLACK_Py.Data - RED_Py.Data;
RelDist_center = sqrt(dx.^2 + dy.^2);


% --------- Relative position: LAR vs Right Camera ---------
dx_cam = LAR_Px - r_CAM_Px;
dy_cam = LAR_Py - r_CAM_Py;
r_RelDist_LARCam = sqrt(dx_cam.^2 + dy_cam.^2);

% --------- Relative position: LAR vs Left Camera ---------
dx_cam = LAR_Px - l_CAM_Px;
dy_cam = LAR_Py - l_CAM_Py;
l_RelDist_LARCam = sqrt(dx_cam.^2 + dy_cam.^2);

% --------- Relative Angle: LAR vs Cameras (both direction vectors point directly from chaser, aligned with its direction) ---------
RelAngle = min(abs(BLACK_Rz.Data - RED_Rz.Data), 2*pi - abs(BLACK_Rz.Data - RED_Rz.Data));

% --------- Build table ---------
data = table(t, ...
             RelDist_center, r_RelDist_LARCam,l_RelDist_LARCam, RelAngle, ...
             'VariableNames', {'Time_s', ...
                               'RelDist_center_m','RelDist_LARCam_right','RelDist_LARCam_left','RelAngle'});

% --------- Write CSV ---------
writetable(data, outputFile);
fprintf('Relative pose data written to: %s\n', outputFile);


% --------- Plot Rel Dist and Angles ---------
tiledlayout(2,1)

% Top plot
nexttile
plot(t,r_RelDist_LARCam)
hold on
plot(t,l_RelDist_LARCam)
plot(t, RelDist_center)
hold off
xlabel("Time (s)")
ylabel("Distance (m)")
title('Relative Distance')
legend("LAR-CAM (right)", "LAR-CAM (left)","Center-to-Center")

% Bottom plot
nexttile
plot(t,RelAngle*180/pi)
title('Relative Angle')
xlabel("Time (s)")
ylabel("Angle (deg)")
