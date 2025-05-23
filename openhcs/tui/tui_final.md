
__________________________________________________________________________________________
| |_[_Global_Settings_]_[_Help]_|_OpenHCS_V1.0____________________________________________
| |_____________1_plate_manager_____________________|_|_________2_Pipeline_editor________|
|_|[add]_[del]_[edit]_[init]_[compile]_[run]________|_|_[add]_[del]_[edit]_[load]_[save]_|
|o| ^/v 1: axotomy_n1_03-02-24 | (...)/nvme_usb/    |o| ^/v 1: pos_gen_pattern           | 
|!| ^/v 2: axotomy_n2_03-03-24 | (...)/nvme_usb/    |o| ^/v 2: enhance + assemble        | 
|?| ^/v 3: axotomy_n3_03-04-24 | (...)/nvme_usb/    |o| ^/v 3: trace analysis         | 
| |                                                 | |                                  | 
| |                                                 | |                                  | 
| |                                                 | |                                  | 
| |                                                 | |                                  | 
| |                                                 | |                                  | 
| |                                                 | |                                  | 
| |                                                 | |                                  | 
|_|________________________________________________ |_|___________________________________
|_Status:_..._______________________________________|_____________________________________|


VER YVERY CLOSE.
I realize my mental model is not yet complete. let me crystilaie it for you
 I will not let you code until I say you are perfectly alighed.

this is the new main menu Canonical for now. It will require a change in our metnal model too I understand. bnut this should make more sense.
top horizontal  bar ->
 global setting open a window for static introspection of defautl config and allows it to be changed
Help open a winow with text and an ok button for now
2nd horizontal bar ->
Title for plate manager window and bar split into ohter bar for piepline editor (step viwer renamed)
3rd horizontal  bar ->
buttons for the menu for the platmangere and pipeline editor
list pane left: Plate (orechesatror) list. 
lsit pane right: list of steps (which make the pipeilne) 

detrailed functoin:
3rd horizontal bar buttons
platemanager:
add: make an orchestrtor wit hthe default config (as per global settings) using a filepath obtained through file selesct dialog
multiple folders may be slected at once.
this then adds a new entry in the list of plates (orchestrators below the bar) and also updated the 
veritcal symbol bar on the left to hvae teh ? syumbol meaning not initialized yet
del: removes the currently seleted plate(s)
edit: create a new custom config for the slected plates . it reflects the current gloabl settings and allows the parmeters to be edited and then saved in a new config just forteh slected plates. very similar logic in gui sltatic relflection as in funciton pattern eiditor
init: this makes each slected plate (orchestrator) run thei intialize() method. if no errors then upate with a yellow - symbol  in the symobol on its left vertical bar. this indicates it is initialized but not compiled 
compile:
this runs the pipeline compiler using the list of steps 
__________________________________________________________________________________________
| |_[_Global_Settings_]_[_Help]_|_OpenHCS_V1.0___________________________________________|
| |_____________1_plate_manager______________________|_|__2_Pipeline_editor______________|
|_|[add]_[del]_[edit]_[init]_[compile]_[run]_________|_|[add]_[del]_[edit]_[load]_[save]_|
|o| ^/v 1: axotomy_n1_03-02-24 | (...)/nvme_usb /    |o| ^/v 1: pos_gen_pattern          | 
|!| ^/v 2: axotomy_n2_03-03-24 | (...)/nvme_usb/     |o| ^/v 2: enhance + assemble       | 
|?| ^/v 3: axotomy_n3_03-04-24 | (...)/nvme_usb/     |o| ^/v 3: trace analysis           | 
| |                                                  | |                                 | 
| |                                                  | |                                 | 
| |                                                  | |                                 | 
| |                                                  | |                                 | 
| |                                                  | |                                 | 
|_|_________________________________________________ |_|_________________________________|
|_Status:_...________________________________________|_|_________________________________|

Dual STEP/FUNC editor sytem:
when clicking edit on a step in the pipeline editor, et replaces teh plate_manager window with the following:

a scrollable pane with a menu bar with two bottons to toggle th viewable menu. 
one for abstract step parameeters (all optional)
one for generated the func pattern required for func step (implements abastract step)
this pane returns takes a funcstep to intialize itself by populating everything it needs statically
it uses all the params of a funcstep, including those inherited by abstract step
one toggleable menu for abstract step params
another toggle menu for the only funcstep param (can be callable enahnce (calalble or (callable , dict kwargs), lsit of callabled enhaned, dict of any combinatoin of the previous)
save updates the step by assigning it a new step constructed with all params collected from both toggleable menus to make a full funcstep
it is greayed out unless any change is made the nit becomes available. it gets greyed out after clicking saved, and ungreyed again if soemthign cahnges .
both panes shuold be identical but just be populated using inspection of diffferent pobjects but still be statitc insepction for dynamic ui
the close buttoon makes teh func/step menu disappear and allow hte plate manger to take its place back

STEP/FUNC Menu 
Step Menu
all these confiruabable items are optional params in abastract step
_________________________________________________
|_X_Step_X_|___Func___|_[save]__[close]__________| <- func/step menu bar
|^|_Step_settngs_editor__[load]_[save_as]________| <- step settings editor with load pickled .step step object or save as new using file dialog
|X| [reset] Name: [...]                          | <- give step a name. can reset to set to None
|X| [reset] input_dir: [...]                     | <- open file select diralog in orchesartor plate_path
|X| [reset] output_dir: [...]                    | <- open file select diralog in orchesartor plate_path
|X| [reset] force_disk_ouput: [ ]                | <- checkbox for true or false 
|X| [reset] variable_components:  |site|V|       | <- default enum DEFAULT_VARIABLE_COMPONENTS and lists VariableComponents enum class names associated to their value and  
|X| [reset] group_by:  |channel|V|               | <- default enum DEFAULT_GROUP_BY and lists GroupBy enum class names associated to their value and  
|X|                                              |    all these confiruable items are optional params in abastract step
|X|                                              |
|X|                                              |
|X|                                              |
|X|                                              |
|X|                                              |
|X|______________________________________________|


Func Menu
all these confiruabable items are optional params in abastract step
_________________________________________________
|___Step___|_X_Func_X_|_[save]__[close]_________| <- func/step menu bar
|^|_Func_Pattern_editor_[add]_[load]_[save_as]__| <- func pattern settings editor with load pickled .func func pattern object or save as new using file dialog
|X|_dict_keys:_|None|V|_+/-__|__[edit_in_vim]_?_| <- you can have more than one key, making he list of funcs change.
|X| Func 1: |percentile stack normalize|v|      | <- drop down menu generated from using the func register and static analysis of func name definition)
|X| --------------------------------------------|
|X|  move  [reset] percentile_low:  0.1 ...     | <- these two kwargs with editabel fields are autogen from the func definition)
|X|   /\   [reset]  percentile_high: 99.9 ...   |
|X|   \/   [add]                                |
|X|        [delete]                             |
|X| ____________________________________________|
|X| Func 2: |n2v2|V|                            |
|X|---------------------------------------------|
|X|         [reset] random_seed: 42 ...         |<- so are these
|X|   move  [reset] device: cuda ...            |
|X|    /\   [reset] blindspot_prob: 0.05 ...    |
|X|    \/   [reset] max_epochs: 10 ...          |
|X|         [reset] batch_size: 4 ...           |
|X|         [reset] patch_size: 64 ...          |
|X|         [reset] learning_rate: 1e-4 ...     |
|X|         [reset] save_model_path: ...        |
|X|         [reset all]                         |
|V|         [add]                               |
| |         [delete]	                        |
| | ____________________________________________|
| | Func 3: |3d_deconv|V|                       |
| |         [reset] random_seed:  42 ...        |
| |    move [reset] device: cuda  ...           |
| |     /\  [reset] blindspot_prob: 0.05 ...    |
|_| vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv|






errors:

The right most main menu will have:
add plate
add step

Select a storage backend from a dropdown (e.g., "Local Disk", "Memory", "Zarr Store", "OperaPhenix"). This selection is mandatory.

Almost true but not quite. disk memory and zarr are storage backends. their selection is not necessary because there are defaults for them in the config. you can cahnge the defautls per orcehstrator by havging the plate be selected and go in settings and ahve static refelction of the config. 

Operaphenix and Iamgexpress are microscope types that are supported. you can manually set it in teh settings I just descibed but by default it is on auto. eventually there will be a feature to add yoour own microsopes. 

Validation: Incorreect. the tui validates when the pre-compile button is pressed and the palte is selected in the platemanger pane. this runs initialization and rasises any error as a message error with an ok box and jsut goes back to the unitialized state and the palte is visually marked with an error mesage which can be printed in the bar at the bottom
