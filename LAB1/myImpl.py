import util

"""
Data sturctures we will use are stack, queue and priority queue.

Stack: first in last out
Queue: first in first out
    collection.push(element): insert element
    element = collection.pop() get and remove element from collection

Priority queue:
    pq.update('eat', 2)
    pq.update('study', 1)
    pq.update('sleep', 3)
pq.pop() will return 'study' because it has highest priority 1.

"""

"""
problem is a object has 3 methods related to search state:

problem.getStartState()
Returns the start state for the search problem.

problem.isGoalState(state)
Returns True if and only if the state is a valid goal state.

problem.getChildren(state)
For a given state, this should return a list of tuples, (next_state,
step_cost), where 'next_state' is a child to the current state,
and 'step_cost' is the incremental cost of expanding to that child.

"""
def myDepthFirstSearch(problem):
    visited = {}
    frontier = util.Stack()

    frontier.push((problem.getStartState(), None))

    while not frontier.isEmpty():
        state, prev_state = frontier.pop()

        if problem.isGoalState(state):
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]

        if state not in visited:
            visited[state] = prev_state

            for next_state, step_cost in problem.getChildren(state):
                frontier.push((next_state, state))

    return []

def myBreadthFirstSearch(problem):
    # YOUR CODE HERE
    # util.raiseNotDefined()
    visited = {}
    frontier = util.Queue()

    frontier.push((problem.getStartState(), None))

    while not frontier.isEmpty():
        state, prev_state = frontier.pop()

        if problem.isGoalState(state):
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]

        if state not in visited:
            visited[state] = prev_state
            # print(state)
            for next_state, step_cost in problem.getChildren(state):
                frontier.push((next_state, state))

    return []

def myAStarSearch(problem, heuristic):
    # YOUR CODE HERE
    # util.raiseNotDefined()
    visited = {}
    g_score = {}

    frontier = util.PriorityQueue()

    frontier.update((problem.getStartState(),None),0)

    g_score[problem.getStartState()] = 0

    while not frontier.isEmpty():
        state,prev_state = frontier.pop()
        current_g_score = g_score[state]

        if problem.isGoalState(state):
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]

        if state not in visited:
            visited[state] = prev_state
            # print(prev_state)
            for next_state, step_cost in problem.getChildren(state):
                g_n = current_g_score + step_cost
                h_n = heuristic(next_state)
                # add children g_score
                g_score[next_state] = g_n
                # f(n) = g(n) + h(n)
                frontier.update((next_state,state),g_n+h_n)

    return []

"""
Game state has 4 methods we can use.

state.isTerminated()
Return True if the state is terminated. We should not continue to search if the state is terminated.

state.isMe()
Return True if it's time for the desired agent to take action. We should check this function to determine whether an agent should maximum or minimum the score.

state.getChildren()
Returns a list of legal state after an agent takes an action.

state.evaluateScore()
Return the score of the state. We should maximum the score for the desired agent.

"""
class MyMinimaxAgent():

    def __init__(self, depth):
        self.depth = depth

    def minimax(self, state, depth):
        if state.isTerminated():
            return state, state.evaluateScore()
        if depth == 0:
            return state, state.evaluateScore()

        best_state, best_score = None, -float('inf') if state.isMe() else float('inf')

        for child in state.getChildren():
            # YOUR CODE HERE
            # util.raiseNotDefined()
            if child.isMe():
                # ??????????????????
                res_state,res_score = self.minimax(child,depth-1)
                if best_score > res_score:
                    best_state = child
                    best_score = res_score
            elif state.isMe():
                # ????????? -- max
                res_state,res_score = self.minimax(child,depth)
                if best_score < res_score:
                    best_state = child
                    best_score = res_score
            else:
                # ????????????????????????????????????????????????????????? -- min
                res_state, res_score = self.minimax(child, depth)
                if best_score > res_score:
                    best_state = child
                    best_score = res_score

        return best_state, best_score

    def getNextState(self, state):
        best_state, _ = self.minimax(state, self.depth)
        return best_state

class MyAlphaBetaAgent():

    def __init__(self, depth):
        self.depth = depth

    def AlphaBeta(self,state,depth,a,b):

        if depth == 0 or state.isTerminated():
            return state, state.evaluateScore()

        best_state, best_score = None, -float('inf') if state.isMe() else float('inf')

        for child in state.getChildren():
            if child.isMe():
                # ?????????????????? -- min
                temp_state, temp_score = self.AlphaBeta(child, depth-1,a,b)
                if best_score > temp_score:
                    best_state = child
                    best_score = temp_score
                # b = min(b, best_score)
                if best_score < a:
                    # ?????? ??? a == b ??????????????????????????????
                    break
                b = min(b, best_score)

            elif state.isMe():
                # ????????? -- max
                temp_state, temp_score = self.AlphaBeta(child, depth, a, b)
                if best_score < temp_score:
                    best_state = child
                    best_score = temp_score
                # a = max(a, best_score)
                if best_score > b:
                    # ?????? ??? a == b ??????????????????????????????
                    break
                a = max(a, best_score)
            else:
                # ????????????????????????????????????????????????????????? -- min
                temp_state, temp_score = self.AlphaBeta(child, depth, a, b)
                if best_score > temp_score:
                    best_state = child
                    best_score = temp_score
                # b = min(b, best_score)
                if best_score < a:
                    # ?????? ??? a == b ??????????????????????????????
                    break
                b = min(b, best_score)

        return best_state, best_score

    def getNextState(self, state):
        # YOUR CODE HERE
        # util.raiseNotDefined()
        a = -float('inf')
        b = float('inf')
        best_state, _ = self.AlphaBeta(state, self.depth,a,b)
        return best_state
