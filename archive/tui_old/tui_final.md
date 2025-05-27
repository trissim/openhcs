
main menu:
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

I will refer by bars by their positoin globablly but describe encapsulation

1. top horizontal  bar -> buttons for global settings, help and exit with OpenHCS_V1.0 in the middle
Help open a winow with text and an ok button for now
Enscapsulate start - 
Then we have a V split with two panes that have this same pattern of layout
2. 2nd horizontal bar ->
Title bars for each menu. left menu is plate manager, right is pipeline editor (step viewer renamed)
3. 3rd horizontal  bar ->
button bars for each menu. left buttons are for plate manager, right are for pipeline editor
area under .
4. interactive list pane -> left: list of plates (orchestrators), right : lsit of steps (which make the pipeline)
the items are clicable and selectablee, and multi selectable.
having a list items selected allow commands from the button bar to run on them
there is a scrollbar that appears when needed.
it drops down to the vey end of the screen to connect with the bottom status bar
items can be moved up and down. they can also be reordered by clicking the ^ and v buttons.
Encapsulate end - 
5. bottom horizontal bar ->
status bar with live logger output (toggleable via `y` in Settings).
clicking the status bar expands a bottom drawer overlay with full log history.

buttons for 1. top horiz bar -> 
 Global settings: show reflection of config.py gloabalpipelineconfig object constructor signature.
                     open a window where each key is a label and there is a field to edit the value.
                     all validation of all fiels is taken care whe nthe compile button is pressed in the plate manager.
help: open a window with text and an ok button for now
exit: exit the app

buttons for 3. 3rd horizontal bar ->
plate manager:
add: make an orchestrtor wit hthe default config (as per global settings) using a filepath obtained through file selesct dialog
multiple folders may be slected at once.
this then adds a new entry in the list of plates (orchestrators below the bar) and also updated the 
veritcal symbol bar on the left to have appropriate syumbol meaning not initialized yet
del: removes the currently seleted plate(s)
init: this makes each slected plate (orchestrator) run thei intialize() method. if no errors then upate with a yellow - symbol  in the symobol on its left vertical bar. this indicates it is initialized but not compiled 
edit: create a new custom config for the slected plates . this shuold just open the same menu as global config but it saves an override for those specific plates (orcehestrators)
compile:
this runs the pipeline compiler using the list of steps associated with each orchestrator that are seelcted. if none selected, it automatically sequentially compiles all the plates that have been initialized successfully. if there are any errors, it stops and displays the error in the status bar. if no errors, it updates the vertical symbol bar on the left to have a green check mark. this indicates it is compiled and ready to run.
run: this runs the pipeline for the selected plates. if none selected, it automatically sequentially runs all the plates that have been compiled successfully. if there are any errors, it stops and displays the error in the status bar. if no errors, it updates the vertical symbol bar on the left to have a green check mark. this indicates it is compiled and ready to run.

pipeline editor:
add: add a new step to the pipeline. it adds a new entry in the list of steps (below the bar) with nothing passed in the constructor. 
del: remove the selected step(s) from the pipeline
edit: open the dual step/func editor to configure the selected step(s) see furhter down for details V
load: load a pipeline definition from a .pipeline file. this replaces the current pipeline with the one loaded. it does not affect the plates in the plate manager.
save: save the current pipeline to a .pipeline file. it does not affect the plates in the plate manager.
.pipeline files are a list of steps that are pickled and saved to disk. they are loaded and saved using the pickle module.

STEP/FUNC Menu 
This Menu Replaces the Platemanger container and replaces it with the dual step/func editor container when the edit button is clicked in the pipeline editor. This set of 2 menus (toggleable step menu and func menu encapsulated in a container) serves to replace the selected step that was edited with a new one using the params collected from the two menus.
The top bar of this container has 4 buttons. 2 buttons to toggle between the step menu and func menu. 2 buttons to save and close the editor. the save button is disabled until changes are made. the close button closes the editor and returns to the pipeline editor.

Fields are populated using static analysis of constructor inspection to obtain labels from keys and default values from the signature. This is done in both the Step menu using abstract step and in the func menu using static analysis of the functions kwargs for visually building and editing a func pattern object by selecting functoins from a dropdown for each func in the list of functions.

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

A funcstep takes a single required param, the func pattern. The func pattern can be a callable, a tuple of (callable, dict_kwargs), a list of callables, or a dict of list of callables. This patternization is represented through the func pattern editor, allowing it to return a singel funcpattern object that is passed to the funcstep required param for funcstep construction. The func pattern editor allows you to build this pattern. It has a list of functions that are available to use, obtained from the function register taht is dynamically populated at runtime from the processing folder based on the decorators used. functions can be selected from the register through the dropdown list. All functions only show their kwargs for presenting keys as label and default value as default value in editable field. like in the step menu. The reset buttons set the value back to the inspected signature default. The func pattern editor allows you to build the pattern by adding functions and their kwargs. It also allows you to save and load the pattern to a .func file. The .func file is a pickled func pattern object. and the funcstep for the func menu. the fields are generated from the constructor signature of the funcstep class using static analysis.  
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


