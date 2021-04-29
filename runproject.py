#from MNIST_TTT4275 import mnist as mn
from Iris_TTT4275 import iris_class as ic




def main():
    #could add options to print different informations and/or make it possible to save the images.
    #If you want to change it to save images, just add the option of taking taskx(s=True)
    #with the new python is it possible to make this a switch case or as it is called in python match case, didnt have the version that supported this
    #could make it so that you dont need to type iris or mnist every time, but thats a possible improvement for later.
    run = True
    while run:
        action = str(input('What task do you want to check out?\n For the Iris task type <<Iris>> and for the MNIST task type <<MNIST>> or quit by typing <<quit>> or <<q>> at anytime (its is not case sensitive).\n your choice: ')).lower()
        if action == 'iris':
            task = str(input('Which task do you want to see? We have task 1ac, task 1d, task 2a, task 2b or task 2b1. \n just type the number and letter, ex: <<1ac>> or 2a. Or just run all with <<all>>.\n Your choice: ')).lower()
            if task == 'q' or task == 'quit':
                print('Quitting...')
                print('Goodbye!')
                run = False
                #action = 'q'
            elif task == '1ac':
                print('Running task 1ac...')
                ic.task1a()
            elif task == '1d':
                print('Running task 1d...')
                ic.task1d()
            elif task == '2a':
                print('Running task 2a...')
                ic.task2a()
            elif task == '2b':
                print('Running task 2b...')
                ic.task2b_1()
            elif task == '2b1':
                option = str(input('There are her two options, both models are only using 1 feature.\n Option 1 is only using Petal length and option 2 is only using Petal width. \n Type 1 or 2: ')).lower()
                if option == 'q' or option == 'quit':
                    print('Quitting...')
                    print('Goodbye!')
                    run = False

                elif option == '1':
                    print('Running task 2b1_1...')
                    ic.task2b_2()
                elif option == '2':
                    print('Running task 2b1_2...')
                    ic.task2b_2_1()
                else:
                    print('Wrong input')
                    action = 'iris'
            elif task == 'all':
                print('Running all task...')
                ic.task1a()
                ic.task1d()
                ic.task2a()
                ic.task2b_1()
                ic.task2b_2()
                ic.task2b_2_1()
            else:
                print('Wrong input')
                action = 'iris'
            
        elif action == 'mnist':
            pass

        elif action == 'quit' or action == 'q':
            print('Quitting...')
            print('Goodbye!')
            run = False

        else:
            print('Wrong input, please try again.')


if __name__ == '__main__':
    main()