using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PathManager : MonoBehaviour{

    public Transform car;
    public Transform target;
    public List<Cell> openSet = new List<Cell>();
    public List<Cell> closedSet = new List<Cell>();
    CellManager cellManager;
    TargetManager targetManager;

    void Awake(){

        cellManager = GetComponent<CellManager>();
        targetManager = GetComponent<TargetManager>();

        cellManager.Initialize();

    }

    void Update(){

        FindPath(car.position,target.position);

    }

    void FindPath(Vector3 s, Vector3 e){

        //Clear the previous frame's data
        targetManager.ClearTargets();
        openSet.Clear();
        closedSet.Clear();

        //Set the start and end of the path
        Cell start = cellManager.WorldPosToCell(s);
        Cell end = cellManager.WorldPosToCell(e);

        //Add the start to your open set
        openSet.Add(start);

        //Loop until we've cleared our open set
        while(openSet.Count > 0){

            //Set the current cell to the first cell in the set
            Cell currentCell = openSet[0];

            for(int i = 1; i < openSet.Count;i++){

                //If the open set and closed set have the same F cost then compare their H costs instead (distance to target Cell)
                if(openSet[i].F() < currentCell.F() 
                || (openSet[i].F() == currentCell.F() && openSet[i].h < currentCell.h)){

                                
                    currentCell = openSet[i];

                }

            }

            //Remove the cell from the open set and add it to our closed set
            openSet.Remove(currentCell);
            closedSet.Add(currentCell);

            if(currentCell == end){

                //We've reached our target so retrace our steps and tell the car to start moving towards its target
                Retrace(start,end);
                targetManager.SetCarTarget();
                return;

            }

            //Look through the current cells neighbouring cells and set their G cost, H cost and its parent cell
            foreach(Cell neigh in cellManager.GetNeighbours(currentCell)){

                if(!neigh.walkable || closedSet.Contains(neigh)){

                    continue;

                }

                //Calculate the path distance from the start to the neighbour
                int disToNeighbour = currentCell.g + GetDis(currentCell,neigh);

                //If the G cost of the neighbouring cell is inaccurate then update the cell
                if(disToNeighbour < neigh.g || !openSet.Contains(neigh)){

                    neigh.g = disToNeighbour;
                    neigh.h = GetDis(neigh,end);
                    neigh.parent = currentCell;

                    if(!openSet.Contains(neigh)){

                        //Add the neighbouring cell to the open set
                        openSet.Add(neigh);

                    }

                }

            }

        }


    }

    void Retrace(Cell start,Cell end){

        List<Cell> path = new List<Cell>();

        //Start at the end of the path
        Cell current = end;

        //Loop backwards
        while(current != start){
            path.Add(current);
            //Spawn a target at the Cells position
            targetManager.SpawnTarget(current.pos);

            //Set the current Cell to its parent Cell
            current = current.parent;

        }

        //Reverse the path so that it goes start to end
        path.Reverse();
        targetManager.targets.Reverse();

    }

    int GetDis(Cell a, Cell b){

        int x = Mathf.Abs(a.gridPos.x - b.gridPos.x);
        int y = Mathf.Abs(a.gridPos.y - b.gridPos.y);

        if(x > y){

            return 14 * y + 10 * (x-y);

        } else {

            return 14 * x + 10 * (y-x);

        }
         
    }

}
