#Default Setting

scenario_type = fMRI;
#scenario_type = fMRI_emulation;
#scan_period=1500;
pulses_per_scan = 1; 
pulse_code = 30;

active_buttons = 4;
button_codes = 1,2,3,4; 
#response_logging = log_active;

default_font_size = 40;
default_font = "ＭＳ明朝";
default_background_color = 100, 100, 100; 
default_text_color = 255,255,255;


$fixation_intensity="255,255,255";
$Display_Width = 1280;
$Display_Height = 720;

#SDL=======================================================
begin;

trial{
	trial_mri_pulse=1;
	stimulus_event{
		sound{wavefile{filename="Run1_0.wav";}T1wav;}T1sound;
		time=0;code="Run1_0.wav";}T1event;
	stimulus_event{picture{box{width = 30; height=5;color = 0,0,0;};x=0;y=0;
			box{width = 5; height=30;color = 0,0,0;};x=0;y=0;};deltat = 0;duration=14500;};
}T1;



#PCL=======================================================
begin_pcl;

int runN = 1; #Change this parameter to specify run
int trialN= 41;
int pulse_start = 1;
int pulse_total; 
int TR = 1500;
int i;
int Duration = 15000;
int st;


array <int> pul[trialN];



pul[1] = pulse_start;
loop i = 2 until i>trialN begin;
	pul[i] = pulse_start+(i-1)*Duration/TR; 
	i = i + 1;
end;



loop i = 1 until i > trialN begin;
	T1.set_mri_pulse(pul[i]);
	T1wav.set_filename("Run"+ string(runN)+"_"+string( i - 1 )+".wav");
	T1event.set_event_code("Run"+ string(runN)+"_"+string( i - 1 ));
	T1wav.load();
	st = clock.time();
	T1.present();
	T1wav.unload();
	
	i=i+1;
end;







#------------------------------------
# End Of File
#------------------------------------
